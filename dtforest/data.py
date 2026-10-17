"""Seeded synthetic datasets: classification blobs, interleaved moons and circles,
and a regression function with noise. All generators accept ``random_state`` and
default to a fixed seed so results are reproducible.
"""

import numpy as np


def make_blobs(n_samples=400, centers=((-3.0, 0.0), (3.0, 0.0)),
               cluster_std=1.0, random_state=42):
    """Gaussian blobs; ``centers`` may be an int (placed on a circle) or coords."""
    rng = np.random.RandomState(random_state)
    if isinstance(centers, int):
        angles = np.linspace(0, 2 * np.pi, centers, endpoint=False)
        centers = [(5 * np.cos(a), 5 * np.sin(a)) for a in angles]
    centers = np.asarray(centers, dtype=float)
    n_centers = len(centers)
    counts = [n_samples // n_centers] * n_centers
    for i in range(n_samples - sum(counts)):
        counts[i] += 1
    parts, labels = [], []
    for k, (c, n_k) in enumerate(zip(centers, counts)):
        parts.append(rng.normal(loc=c, scale=cluster_std, size=(n_k, 2)))
        labels.append(np.full(n_k, k))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_moons(n_samples=400, noise=0.15, random_state=42):
    """Two interleaving half-moons (non-linearly separable; trees love this)."""
    rng = np.random.RandomState(random_state)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    theta_out = rng.uniform(0, np.pi, n_out)
    theta_in = rng.uniform(0, np.pi, n_in)
    outer = np.column_stack([np.cos(theta_out), np.sin(theta_out)])
    inner = np.column_stack([1.0 - np.cos(theta_in), 1.0 - np.sin(theta_in) - 0.5])
    X = np.vstack([outer, inner])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_circles(n_samples=400, noise=0.08, factor=0.4, random_state=42):
    """Concentric circles (inner disc vs outer ring)."""
    rng = np.random.RandomState(random_state)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    theta_out = rng.uniform(0, 2 * np.pi, n_out)
    theta_in = rng.uniform(0, 2 * np.pi, n_in)
    r_out = np.ones(n_out)
    r_in = rng.uniform(0, factor, n_in)
    outer = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])
    inner = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])
    X = np.vstack([outer, inner])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_regression(n_samples=400, n_features=4, noise=0.5, random_state=42):
    """Axis-aligned friendly regression: a step, a linear term, a quadratic dip.

    y = 10 * [x1 > 0.5] + 3 * x2 - 2 * x3^2 + noise, x4 pure noise feature.
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, n_features))
    y = (10.0 * (X[:, 0] > 0.5)
         + 3.0 * X[:, 1]
         - 2.0 * X[:, 2] ** 2
         + rng.normal(scale=noise, size=n_samples))
    return X, y
