"""Tests for the CART decision-tree implementation."""

import numpy as np
import pytest

from dtforest.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dtforest.data import make_blobs, make_regression
from dtforest.metrics import accuracy_score, r2_score


def test_classifier_perfect_on_separable_blobs():
    X, y = make_blobs(n_samples=200, centers=((-5.0, 0.0), (5.0, 0.0)),
                      cluster_std=0.5, random_state=1)
    tree = DecisionTreeClassifier(random_state=1).fit(X, y)
    assert accuracy_score(y, tree.predict(X)) == 1.0


def test_entropy_criterion_also_separates():
    X, y = make_blobs(n_samples=200, centers=((-5.0, 0.0), (5.0, 0.0)),
                      cluster_std=0.5, random_state=2)
    tree = DecisionTreeClassifier(criterion="entropy", random_state=2).fit(X, y)
    assert accuracy_score(y, tree.predict(X)) == 1.0


def test_invalid_criterion_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="gini2")


def test_max_depth_respected():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=3)
    tree = DecisionTreeClassifier(max_depth=3, random_state=3).fit(X, y)
    assert tree._max_depth_seen <= 3


def test_min_samples_split_respected():
    X, y = make_blobs(n_samples=120, centers=2, cluster_std=0.8, random_state=4)
    tree = DecisionTreeClassifier(min_samples_split=40, random_state=4).fit(X, y)
    # walk the grown tree: no internal node may have fewer samples than the bound
    def check(node):
        if not node.is_leaf:
            assert node.n_samples >= 40
            check(node.left)
            check(node.right)
    check(tree.tree_)


def test_leaf_impurity_zero_on_pure_node():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 0])
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)
    assert tree.tree_.is_leaf
    assert tree.tree_.impurity == 0.0
    assert tree.tree_.value == 0


def test_feature_importances_sum_to_one():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=5)
    tree = DecisionTreeClassifier(random_state=5).fit(X, y)
    imp = tree.feature_importances_
    assert len(imp) == X.shape[1]
    assert np.isclose(imp.sum(), 1.0)


def test_predict_proba_rows_sum_to_one():
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=6)
    tree = DecisionTreeClassifier(random_state=6).fit(X, y)
    proba = tree.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_raises_before_fit():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict(np.zeros((4, 2)))


def test_regressor_high_r2_on_clean_function():
    X, y = make_regression(n_samples=400, noise=0.05, random_state=7)
    tree = DecisionTreeRegressor(max_depth=8, random_state=7).fit(X, y)
    assert r2_score(y, tree.predict(X)) > 0.99


def test_regressor_leaf_value_is_mean():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    tree = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)
    assert not tree.tree_.is_leaf  # variance > 0, so the root must split
    left = tree.tree_.left
    right = tree.tree_.right
    assert left.is_leaf and right.is_leaf
    # best split is between x=1 and x=2: means are 2.0 and 6.0
    assert np.isclose(left.value, 2.0)
    assert np.isclose(right.value, 6.0)
