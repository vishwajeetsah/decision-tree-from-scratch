"""CART decision trees implemented from scratch on NumPy.

``DecisionTreeClassifier`` splits on Gini impurity or entropy, ``DecisionTreeRegressor``
splits on mean squared error (variance reduction). Splits are binary, axis-aligned,
and found by exhaustively scanning midpoints between sorted feature values — the
classic CART greedy search.
"""

import numpy as np


class _Node:
    """A single node of a grown tree."""

    __slots__ = (
        "feature_idx", "threshold", "left", "right",
        "value", "proba", "impurity", "n_samples",
    )

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None        # leaf prediction (class label or mean)
        self.proba = None        # leaf class distribution (classifier only)
        self.impurity = 0.0
        self.n_samples = 0

    @property
    def is_leaf(self):
        return self.left is None


class _BaseDecisionTree:
    """Shared tree-growing machinery (subclasses supply the task specifics)."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 random_state=None, max_features=None):
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_features = max_features  # None -> all; int / float fraction / "sqrt" / "log2"

    # -- task-specific hooks -------------------------------------------------
    def _impurity(self, y):
        raise NotImplementedError

    def _leaf_value(self, y):
        raise NotImplementedError

    # -- public API ----------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y have different lengths")
        if len(X) < 2:
            raise ValueError("need at least 2 samples to fit a tree")
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.RandomState(self.random_state)
        self._importances = np.zeros(self.n_features_in_)
        self._max_depth_seen = 0
        self._prepare_fit(y)
        self.tree_ = self._grow(X, y, depth=0)
        total = self._importances.sum()
        if total > 0:
            self._importances /= total
        return self

    def _prepare_fit(self, y):
        """Hook for task-specific state (e.g. class labels)."""

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        self._check_fitted()
        return np.array([self._walk(x, self.tree_) for x in X])

    def _check_fitted(self):
        if not hasattr(self, "tree_"):
            raise RuntimeError("tree is not fitted yet — call fit() first")

    @property
    def feature_importances_(self):
        """Normalised total impurity decrease per feature (sums to 1)."""
        self._check_fitted()
        return self._importances.copy()

    # -- tree growing --------------------------------------------------------
    def _grow(self, X, y, depth):
        n = len(X)
        self._max_depth_seen = max(self._max_depth_seen, depth)
        node = _Node()
        node.n_samples = n
        node.impurity = self._impurity(y)
        stop = (
            node.impurity == 0.0
            or n < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if not stop:
            split = self._best_split(X, y)
            if split is not None:
                feature, threshold, gain = split
                mask = X[:, feature] <= threshold
                node.feature_idx = feature
                node.threshold = threshold
                node.left = self._grow(X[mask], y[mask], depth + 1)
                node.right = self._grow(X[~mask], y[~mask], depth + 1)
                # importance: impurity decrease weighted by node size
                child_imp = (
                    node.left.impurity * node.left.n_samples
                    + node.right.impurity * node.right.n_samples
                ) / n
                self._importances[feature] += gain_weight(node.impurity, child_imp, n)
        if node.is_leaf:
            node.value = self._leaf_value(y)
            self._set_leaf_proba(node, y)
        return node

    def _set_leaf_proba(self, node, y):
        """Hook: classifier stores the leaf class distribution."""

    def _walk(self, x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.value

    def _feature_subset(self, n_features):
        """Features considered at one split (random subset if max_features set)."""
        mf = self.max_features
        if mf is None:
            return np.arange(n_features)
        if isinstance(mf, str):
            if mf == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            elif mf == "log2":
                k = max(1, int(np.log2(n_features)))
            else:
                raise ValueError(f"unknown max_features={mf!r}")
        elif isinstance(mf, float):
            if not 0.0 < mf <= 1.0:
                raise ValueError("float max_features must be in (0, 1]")
            k = max(1, int(mf * n_features))
        else:
            k = max(1, int(mf))
        return np.sort(self._rng.choice(n_features, size=min(k, n_features),
                                       replace=False))

    def _best_split(self, X, y):
        """Return (feature, threshold, gain) of the best valid split, or None."""
        raise NotImplementedError


def gain_weight(parent_impurity, child_impurity, n_samples):
    """Impurity decrease weighted by the number of samples at the node."""
    return (parent_impurity - child_impurity) * n_samples


class DecisionTreeClassifier(_BaseDecisionTree):
    """CART classification tree.

    Parameters
    ----------
    criterion : {"gini", "entropy"}
    max_depth : int or None
    min_samples_split : int
    min_samples_leaf : int
    random_state : int or None
    max_features : int, float, "sqrt", "log2" or None (used by random forests)
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, random_state=None, max_features=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        super().__init__(max_depth, min_samples_split, min_samples_leaf,
                         random_state, max_features)

    def _prepare_fit(self, y):
        self.classes_ = np.unique(y)
        self._y_idx = np.searchsorted(self.classes_, y)

    def _impurity(self, y):
        idx = np.searchsorted(self.classes_, y)
        counts = np.bincount(idx, minlength=len(self.classes_)).astype(float)
        p = counts / counts.sum()
        if self.criterion == "gini":
            return 1.0 - float((p ** 2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.where(p > 0, p * np.log2(p), 0.0).sum()
        return float(ent)

    def _leaf_value(self, y):
        idx = np.searchsorted(self.classes_, y)
        counts = np.bincount(idx, minlength=len(self.classes_))
        return self.classes_[int(np.argmax(counts))]  # ties -> smallest label

    def _set_leaf_proba(self, node, y):
        idx = np.searchsorted(self.classes_, y)
        counts = np.bincount(idx, minlength=len(self.classes_)).astype(float)
        node.proba = counts / counts.sum()

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        self._check_fitted()
        out = np.zeros((len(X), len(self.classes_)))
        for i, x in enumerate(X):
            node = self.tree_
            while not node.is_leaf:
                node = node.left if x[node.feature_idx] <= node.threshold else node.right
            out[i] = node.proba
        return out

    def _best_split(self, X, y):
        n, m = X.shape
        parent_imp = self._impurity(y)
        if parent_imp == 0.0:
            return None
        y_idx = np.searchsorted(self.classes_, y)
        k = len(self.classes_)
        total_counts = np.bincount(y_idx, minlength=k).astype(float)
        best_gain, best = -np.inf, None
        for f in self._feature_subset(m):
            xs = X[:, f]
            order = np.argsort(xs, kind="mergesort")
            xs_sorted = xs[order]
            # split only between distinct values
            change = np.flatnonzero(np.diff(xs_sorted) != 0)
            if change.size == 0:
                continue
            onehot = np.zeros((n, k))
            onehot[np.arange(n), y_idx[order]] = 1.0
            cum = np.cumsum(onehot, axis=0)
            left_counts = cum[change]
            n_left = change + 1
            n_right = n - n_left
            valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not np.any(valid):
                continue
            lc = left_counts[valid]
            nl = n_left[valid].astype(float)
            rc = total_counts - lc
            nr = n_right[valid].astype(float)
            pl, pr = lc / nl[:, None], rc / nr[:, None]
            if self.criterion == "gini":
                imp_l = 1.0 - (pl ** 2).sum(axis=1)
                imp_r = 1.0 - (pr ** 2).sum(axis=1)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    imp_l = -np.where(pl > 0, pl * np.log2(pl), 0.0).sum(axis=1)
                    imp_r = -np.where(pr > 0, pr * np.log2(pr), 0.0).sum(axis=1)
            weighted = (nl * imp_l + nr * imp_r) / n
            gains = parent_imp - weighted
            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_gain = float(gains[j])
                thr = (xs_sorted[change[valid][j]] + xs_sorted[change[valid][j] + 1]) / 2.0
                best = (int(f), float(thr), best_gain)
        if best is None or best_gain <= 0.0:
            return None
        return best


class DecisionTreeRegressor(_BaseDecisionTree):
    """CART regression tree: splits minimise mean squared error."""

    def _impurity(self, y):
        y = np.asarray(y, dtype=float)
        return float(np.mean((y - y.mean()) ** 2)) if len(y) else 0.0

    def _leaf_value(self, y):
        return float(np.mean(y))

    def _best_split(self, X, y):
        n, m = X.shape
        y = np.asarray(y, dtype=float)
        parent_imp = self._impurity(y)
        if parent_imp == 0.0:
            return None
        total_sum = y.sum()
        total_sq = float((y ** 2).sum())
        best_gain, best = -np.inf, None
        for f in self._feature_subset(m):
            xs = X[:, f]
            order = np.argsort(xs, kind="mergesort")
            xs_sorted = xs[order]
            ys = y[order]
            change = np.flatnonzero(np.diff(xs_sorted) != 0)
            if change.size == 0:
                continue
            cum_sum = np.cumsum(ys)
            cum_sq = np.cumsum(ys ** 2)
            n_left = change + 1
            n_right = n - n_left
            valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not np.any(valid):
                continue
            pos = change[valid]
            nl = n_left[valid].astype(float)
            nr = n_right[valid].astype(float)
            sum_l, sq_l = cum_sum[pos], cum_sq[pos]
            sum_r, sq_r = total_sum - sum_l, total_sq - sq_l
            mse_l = np.maximum(sq_l / nl - (sum_l / nl) ** 2, 0.0)
            mse_r = np.maximum(sq_r / nr - (sum_r / nr) ** 2, 0.0)
            weighted = (nl * mse_l + nr * mse_r) / n
            gains = parent_imp - weighted
            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_gain = float(gains[j])
                thr = (xs_sorted[pos[j]] + xs_sorted[pos[j] + 1]) / 2.0
                best = (int(f), float(thr), best_gain)
        if best is None or best_gain <= 0.0:
            return None
        return best
