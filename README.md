# decision-tree-from-scratch

CART decision trees and random forests implemented on pure NumPy — no
scikit-learn, no black boxes. `DecisionTreeClassifier` splits on Gini impurity
or entropy, `DecisionTreeRegressor` splits on mean squared error, and the
forests wrap them with bootstrap sampling, random feature subsets at every
split, and out-of-bag (OOB) validation.

## What it does

- **CART classification trees** — binary, axis-aligned splits found by scanning
  every midpoint between sorted feature values; Gini or entropy criterion.
- **CART regression trees** — same machinery, splitting on variance reduction.
- **Random forests** — bagged trees with per-split feature subsampling
  (`max_features="sqrt"` / `"log2"` / int / fraction), majority vote for
  classification, mean for regression.
- **Out-of-bag scoring** — each sample is validated against only the trees that
  never saw it, so you get a test-set-like score for free.
- **Feature importances** — normalised total impurity decrease per feature.
- **Seeded synthetic data** — blobs, interleaving moons, concentric circles,
  and an axis-aligned regression function with a noise feature.

## Quickstart

```bash
pip install -r requirements.txt
python cli.py --task classify --trees 50 --depth 8 --seed 7
python cli.py --task regress --show-tree
pytest   # 30 tests
```

Use it as a library:

```python
from dtforest.data import make_moons
from dtforest.forest import RandomForestClassifier
from dtforest.metrics import accuracy_score
from dtforest.utils import train_test_split

X, y = make_moons(n_samples=800, noise=0.15, random_state=7)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=7)
forest = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=7)
forest.fit(Xtr, ytr)
print(accuracy_score(yte, forest.predict(Xte)))   # 0.9900
print(forest.oob_score_)                          # 0.9867
print(forest.feature_importances_)                # [0.4212 0.5788]
```

## How it works

**Trees (CART).** At each node the tree evaluates every feature (or a random
subset, when called from a forest): it sorts the feature values, tries every
midpoint between distinct values as a candidate threshold, and picks the split
with the largest impurity decrease. Classification uses
Gini = 1 − Σ p² or entropy = −Σ p log p; regression uses variance
reduction on the target. Growth stops at a pure node, `max_depth`, or when
fewer than `min_samples_split` samples remain; each side of a split must hold
at least `min_samples_leaf` samples.

**Forests (bagging).** Each of the N trees trains on a bootstrap resample of
size n drawn with replacement, and sees only a random feature subset at each
split. A classifier averages the trees' class probabilities and takes the
argmax (majority vote); a regressor averages the predictions. Because each
bootstrap leaves out ~37% of the data, every sample has "out-of-bag" trees
that never trained on it — aggregating those gives the OOB score, a validation
estimate with no held-out set. Feature importance is the mean over trees of
the normalised per-node impurity decrease.

## Measured results (seed 7, n = 800, depth 8, 50 trees)

Classification — interleaving moons, 75/25 split:

| model | test accuracy | F1-macro | OOB |
|---|---|---|---|
| single tree | 0.9800 | — | — |
| random forest | 0.9900 | 0.9899 | 0.9867 |

Regression — `y = 10·[x0>0.5] + 3·x1 − 2·x2² + noise(0.5)`, 75/25 split:

| model | test R² | test MSE | OOB R² |
|---|---|---|---|
| single tree | 0.9514 | 1.7038 | — |
| random forest | 0.9748 | 0.8827 | 0.9823 |

Feature importances (regression forest): x0 (step) 0.5564, x1 (linear) 0.3306,
x2 (quadratic) 0.1112, x3 (noise) 0.0018 — the noise feature is correctly
ranked near zero.

## Honest limitations

- **Split search is O(n·m) per node** (sort + scan over every feature). It is a
  teaching implementation: comfortable at thousands of samples, not millions.
  sklearn uses histogram binning instead.
- **No categorical support.** Everything is treated as a continuous numeric
  feature — there are no encoders, no ordered-category splits.
- **Pruning is depth-based only.** There is no cost-complexity (ccp) pruning,
  no `min_impurity_decrease`; an unpruned tree will happily memorise noise.
- **Trees are piecewise constant.** A regression tree needs many leaves to
  approximate smooth curves — depth 6 reaches R² 0.98 on the demo function,
  depth 8 reaches 0.996.
- **OOB can be optimistic or pessimistic** relative to a true held-out set;
  it is an estimate, not a guarantee.
- Bootstrap resamples are drawn with `np.random.RandomState`, not the newer
  `np.random.Generator` API.

## Project structure

```
dtforest/
  tree.py      DecisionTreeClassifier (Gini/entropy), DecisionTreeRegressor (MSE)
  forest.py    RandomForestClassifier / RandomForestRegressor, OOB scoring
  metrics.py   accuracy, confusion matrix, precision/recall/F1, MSE, R²
  data.py      seeded generators: blobs, moons, circles, regression
  utils.py     train_test_split with seed
cli.py         end-to-end demo: single tree vs forest, ASCII tree rendering
tests/         30 pytest tests covering trees, forests, metrics, data, CLI
```

## What I Learned

- Writing the split search by hand makes bagging's value visceral: a single
  depth-8 tree on noisy moons scores 0.98, but the forest's OOB score (0.9867)
  told me that *before* I ever touched the test set — that free validation
  estimate is the quiet superpower of random forests.
- The trickiest real bug was not in splitting but in bookkeeping: a bootstrap
  resample can miss a rare class entirely, so each tree's `predict_proba`
  columns had to be realigned to the global class labels before averaging.
  Ensemble code fails at the seams between members, not inside them.
- Vectorising the split search with cumulative one-hot counts turned an
  O(n²) inner loop into an O(n log n) sort plus linear scans — the 50-tree
  classification forest trains on 800 samples in ~0.4s (the regression forest,
  with 4 features to scan per split, takes ~2.7s).
- Feature importance fell out naturally from tracking impurity decrease per
  node, and the noise feature scoring 0.0018 vs 0.5564 for the real signal is
  the kind of sanity check that makes you trust the machinery.
- Trees memorise smooth functions slowly: depth-6 regression R² 0.9795 vs
  depth-10's 0.9994 on the same clean function — a concrete feel for the
  depth/capacity trade-off that `max_depth` controls.
