"""Evaluation metrics implemented from scratch."""

import numpy as np


def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Rows are true labels, columns are predicted labels."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def _prf_per_class(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1


def _prf_reduced(y_true, y_pred, average):
    """Per-class precision/recall/F1 reduced by `average`; returns (p, r, f1)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precision, recall, f1 = _prf_per_class(y_true, y_pred, labels)
    if average == "binary":
        if len(labels) != 2:
            raise ValueError("binary average needs exactly 2 classes")
        # positive class = the larger label, mirroring the sklearn convention
        return float(precision[1]), float(recall[1]), float(f1[1])
    if average == "macro":
        return float(precision.mean()), float(recall.mean()), float(f1.mean())
    if average == "micro":
        # for single-label problems micro precision == micro recall == accuracy
        acc = accuracy_score(y_true, y_pred)
        return acc, acc, acc
    raise ValueError(f"unknown average={average!r}; use 'binary', 'macro' or 'micro'")


def precision_score(y_true, y_pred, average="binary"):
    return _prf_reduced(y_true, y_pred, average)[0]


def recall_score(y_true, y_pred, average="binary"):
    return _prf_reduced(y_true, y_pred, average)[1]


def f1_score(y_true, y_pred, average="binary"):
    return _prf_reduced(y_true, y_pred, average)[2]


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)
