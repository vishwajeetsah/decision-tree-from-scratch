"""Random forests: bagged ensembles of CART trees with out-of-bag evaluation.

Each tree trains on a bootstrap resample of the data and considers only a random
subset of features at every split. Classifier prediction is majority vote,
regressor prediction is the mean. Samples left out of a tree's bootstrap
("out-of-bag") give a built-in validation score without a separate split.
"""

import numpy as np

from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class _BaseRandomForest:
    _tree_class = None

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True,
                 oob_score=False, random_state=None, criterion="gini"):
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.criterion = criterion

    def _check_fitted(self):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("forest is not fitted yet — call fit() first")

    def _make_tree(self, seed):
        kwargs = dict(max_depth=self.max_depth,
                      min_samples_split=self.min_samples_split,
                      min_samples_leaf=self.min_samples_leaf,
                      random_state=seed,
                      max_features=self.max_features)
        if self._tree_class is DecisionTreeClassifier:
            kwargs["criterion"] = self.criterion
        return self._tree_class(**kwargs)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        n, m = X.shape
        if len(y) != n:
            raise ValueError("X and y have different lengths")
        if n < 2:
            raise ValueError("need at least 2 samples to fit a forest")
        self.n_features_in_ = m
        master = np.random.RandomState(self.random_state)
        self.estimators_ = []
        oob_state = self._init_oob(n, y)
        for _ in range(self.n_estimators):
            seed = master.randint(0, 2 ** 31 - 1)
            if self.bootstrap:
                idx = master.choice(n, size=n, replace=True)
                in_bag = np.zeros(n, dtype=bool)
                in_bag[np.unique(idx)] = True
                oob_idx = np.flatnonzero(~in_bag)
            else:
                idx = np.arange(n)
                oob_idx = np.array([], dtype=int)
            tree = self._make_tree(seed).fit(X[idx], y[idx])
            self.estimators_.append(tree)
            if self.oob_score and len(oob_idx):
                self._accumulate_oob(tree, X[oob_idx], oob_idx, oob_state)
        importances = np.mean([t.feature_importances_ for t in self.estimators_],
                              axis=0)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        if self.oob_score:
            self.oob_score_ = self._oob_score(oob_state, y)
        return self

    # OOB hooks (implemented per task)
    def _init_oob(self, n, y):
        raise NotImplementedError

    def _accumulate_oob(self, tree, X_oob, oob_idx, state):
        raise NotImplementedError

    def _oob_score(self, state, y):
        raise NotImplementedError

    def _single_preds(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return [t.predict(X) for t in self.estimators_]


class RandomForestClassifier(_BaseRandomForest):
    """Random forest classifier (majority vote of tree probabilities)."""

    _tree_class = DecisionTreeClassifier

    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                 bootstrap=True, oob_score=False, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split,
                         min_samples_leaf, max_features, bootstrap,
                         oob_score, random_state, criterion)

    def _init_oob(self, n, y):
        self.classes_ = np.unique(y)
        return {"prob": np.zeros((n, len(self.classes_))),
                "counts": np.zeros(n, dtype=int)}

    def _align_proba(self, tree, X):
        """Map one tree's predict_proba onto the global class columns.

        A bootstrap resample can miss a rare class, so a tree's columns are
        realigned (zeros for missing classes) before averaging/voting.
        """
        probs = tree.predict_proba(X)
        if len(tree.classes_) == len(self.classes_):
            return probs
        full = np.zeros((len(X), len(self.classes_)))
        cols = np.searchsorted(self.classes_, tree.classes_)
        full[:, cols] = probs
        return full

    def _accumulate_oob(self, tree, X_oob, oob_idx, state):
        state["prob"][oob_idx] += self._align_proba(tree, X_oob)
        state["counts"][oob_idx] += 1

    def _oob_score(self, state, y):
        mask = state["counts"] > 0
        if not np.any(mask):
            return float("nan")
        preds = self.classes_[np.argmax(state["prob"][mask], axis=1)]
        return float(np.mean(preds == y[mask]))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        self._check_fitted()
        return np.mean([self._align_proba(t, X) for t in self.estimators_], axis=0)

    def predict(self, X):
        self._check_fitted()
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class RandomForestRegressor(_BaseRandomForest):
    """Random forest regressor (mean of tree predictions)."""

    _tree_class = DecisionTreeRegressor

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, bootstrap=True,
                 oob_score=False, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split,
                         min_samples_leaf, max_features, bootstrap,
                         oob_score, random_state)

    def _init_oob(self, n, y):
        return {"sums": np.zeros(n), "counts": np.zeros(n, dtype=int)}

    def _accumulate_oob(self, tree, X_oob, oob_idx, state):
        state["sums"][oob_idx] += tree.predict(X_oob)
        state["counts"][oob_idx] += 1

    def _oob_score(self, state, y):
        mask = state["counts"] > 0
        if not np.any(mask):
            return float("nan")
        y = np.asarray(y, dtype=float)
        preds = state["sums"][mask] / state["counts"][mask]
        ss_res = float(((y[mask] - preds) ** 2).sum())
        ss_tot = float(((y[mask] - y[mask].mean()) ** 2).sum())
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict(self, X):
        return np.mean(self._single_preds(X), axis=0)
