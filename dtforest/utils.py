"""Small helpers shared across the package."""

import numpy as np


def train_test_split(X, y, test_size=0.25, random_state=None):
    """Shuffle and split into train/test; ``test_size`` is a fraction."""
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y have different lengths")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be a fraction in (0, 1)")
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
