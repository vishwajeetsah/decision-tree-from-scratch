"""Tests for metrics, data generators, utils, and the CLI smoke test."""

import subprocess
import sys
import os

import numpy as np
import pytest

from dtforest.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score,
)
from dtforest.data import make_blobs, make_moons, make_circles, make_regression
from dtforest.utils import train_test_split

REPO = os.path.join(os.path.dirname(__file__), "..")


def test_accuracy_score():
    assert accuracy_score([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_confusion_matrix_known_case():
    cm = confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1])
    assert cm.tolist() == [[1, 1], [1, 1]]
    assert confusion_matrix([0, 1], [0, 1]).diagonal().tolist() == [1, 1]


def test_precision_recall_f1_binary():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    # TP=2, FP=1, FN=1 -> P=R=F1=2/3 for the positive class
    assert np.isclose(precision_score(y_true, y_pred), 2 / 3)
    assert np.isclose(recall_score(y_true, y_pred), 2 / 3)
    assert np.isclose(f1_score(y_true, y_pred), 2 / 3)
    with pytest.raises(ValueError):
        f1_score([0, 1, 2], [0, 1, 2])  # binary average on 3 classes


def test_f1_macro():
    # class 0: P=2/3, R=1 -> F1=0.8; class 1: P=1, R=1/2 -> F1=2/3
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 0]
    assert np.isclose(f1_score(y_true, y_pred, average="macro"), 11 / 15)


def test_regression_metrics():
    assert mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    # mean predictor -> R2 == 0
    assert np.isclose(r2_score([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]), 0.0)


def test_data_generators_shapes_and_reproducibility():
    X, y = make_blobs(n_samples=100, centers=3, random_state=0)
    assert X.shape == (100, 2) and y.shape == (100,) and len(np.unique(y)) == 3
    X2, y2 = make_blobs(n_samples=100, centers=3, random_state=0)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    Xm, ym = make_moons(n_samples=80, random_state=1)
    assert Xm.shape == (80, 2) and set(np.unique(ym)) == {0, 1}
    Xc, yc = make_circles(n_samples=80, random_state=2)
    assert Xc.shape == (80, 2) and set(np.unique(yc)) == {0, 1}
    Xr, yr = make_regression(n_samples=60, n_features=4, random_state=3)
    assert Xr.shape == (60, 4) and yr.shape == (60,)
    # noise feature x3 should be weakly related to y
    assert abs(np.corrcoef(Xr[:, 3], yr)[0, 1]) < 0.4


def test_train_test_split_disjoint_and_seeded():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=5)
    assert len(Xte) == 6 and len(Xtr) == 14
    assert len(set(ytr) & set(yte)) == 0  # no overlap
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.3, random_state=5)
    assert np.array_equal(Xtr, Xtr2) and np.array_equal(Xte, Xte2)


def test_cli_classify_smoke():
    r = subprocess.run(
        [sys.executable, "cli.py", "--task", "classify", "--n-samples", "200",
         "--trees", "5", "--depth", "4", "--seed", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "test accuracy" in r.stdout and "OOB accuracy" in r.stdout


def test_cli_regress_smoke():
    r = subprocess.run(
        [sys.executable, "cli.py", "--task", "regress", "--n-samples", "200",
         "--trees", "5", "--depth", "4", "--seed", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "test R" in r.stdout and "OOB R" in r.stdout
