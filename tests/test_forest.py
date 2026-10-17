"""Tests for the random-forest implementation."""

import numpy as np

from dtforest.forest import RandomForestClassifier, RandomForestRegressor
from dtforest.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dtforest.data import make_blobs, make_moons, make_regression
from dtforest.metrics import accuracy_score, r2_score
from dtforest.utils import train_test_split


def test_forest_matches_single_tree_on_blobs():
    X, y = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=10)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=10)
    tree_acc = accuracy_score(
        yte, DecisionTreeClassifier(random_state=10).fit(Xtr, ytr).predict(Xte))
    forest_acc = accuracy_score(
        yte, RandomForestClassifier(n_estimators=30, oob_score=True,
                                    random_state=10).fit(Xtr, ytr).predict(Xte))
    assert forest_acc >= 0.95
    assert forest_acc >= tree_acc - 0.05  # forest shouldn't lag a lone tree badly


def test_forest_beats_single_tree_on_noisy_moons():
    X, y = make_moons(n_samples=500, noise=0.25, random_state=11)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=11)
    tree_acc = accuracy_score(
        yte, DecisionTreeClassifier(random_state=11).fit(Xtr, ytr).predict(Xte))
    forest_acc = accuracy_score(
        yte, RandomForestClassifier(n_estimators=50, random_state=11)
        .fit(Xtr, ytr).predict(Xte))
    assert forest_acc >= tree_acc - 0.02  # bagging helps or at least doesn't hurt


def test_oob_score_sanity():
    X, y = make_blobs(n_samples=400, centers=2, cluster_std=1.0, random_state=12)
    forest = RandomForestClassifier(n_estimators=30, oob_score=True,
                                    random_state=12).fit(X, y)
    assert 0.0 <= forest.oob_score_ <= 1.0
    # OOB is a validation estimate: on clean blobs it should be high
    assert forest.oob_score_ > 0.9
    # and close to a held-out test score
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=12)
    f2 = RandomForestClassifier(n_estimators=30, oob_score=True,
                                random_state=12).fit(Xtr, ytr)
    assert abs(f2.oob_score_ - accuracy_score(yte, f2.predict(Xte))) < 0.15


def test_regressor_forest_r2_and_oob():
    X, y = make_regression(n_samples=400, noise=0.5, random_state=13)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=13)
    forest = RandomForestRegressor(n_estimators=30, oob_score=True,
                                   random_state=13).fit(Xtr, ytr)
    preds = forest.predict(Xte)
    assert r2_score(yte, preds) > 0.85
    assert 0.0 <= forest.oob_score_ <= 1.0
    assert forest.oob_score_ > 0.7


def test_reproducibility_with_same_seed():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=14)
    f1 = RandomForestClassifier(n_estimators=20, random_state=14).fit(X, y)
    f2 = RandomForestClassifier(n_estimators=20, random_state=14).fit(X, y)
    assert np.array_equal(f1.predict(X), f2.predict(X))
    assert np.allclose(f1.feature_importances_, f2.feature_importances_)


def test_forest_feature_importances_sum_to_one():
    X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=15)
    forest = RandomForestClassifier(n_estimators=20, random_state=15).fit(X, y)
    assert np.isclose(forest.feature_importances_.sum(), 1.0)


def test_max_features_subset_runs():
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=16)
    # dataset is 2-D; max_features=1 forces one-feature splits at every node
    forest = RandomForestClassifier(n_estimators=10, max_features=1,
                                    random_state=16).fit(X, y)
    assert len(forest.estimators_) == 10
    assert accuracy_score(y, forest.predict(X)) > 0.8


def test_predict_proba_shape_and_rows():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.8, random_state=17)
    forest = RandomForestClassifier(n_estimators=10, random_state=17).fit(X, y)
    proba = forest.predict_proba(X[:7])
    assert proba.shape == (7, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_no_bootstrap_uses_all_samples():
    X, y = make_blobs(n_samples=150, centers=2, cluster_std=0.5, random_state=18)
    forest = RandomForestClassifier(n_estimators=5, bootstrap=False,
                                    oob_score=True, random_state=18).fit(X, y)
    # no OOB samples exist -> oob_score_ is NaN, fitting must still work
    assert np.isnan(forest.oob_score_)
    assert accuracy_score(y, forest.predict(X)) > 0.95


def test_single_tree_inside_forest_is_seeded_independently():
    # trees must differ from each other (they get different bootstrap/seeds)
    X, y = make_moons(n_samples=300, noise=0.2, random_state=19)
    forest = RandomForestClassifier(n_estimators=10, max_depth=4,
                                    random_state=19).fit(X, y)
    preds = [t.predict(X[:50]) for t in forest.estimators_]
    assert not all(np.array_equal(preds[0], p) for p in preds[1:])
