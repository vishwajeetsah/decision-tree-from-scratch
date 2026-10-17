"""Command-line demo: single tree vs random forest on synthetic data.

Usage:
    python cli.py --task classify --n-samples 800 --depth 8 --trees 50 --seed 7
    python cli.py --task regress --show-tree
"""

import argparse
import time

import numpy as np

from dtforest.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dtforest.forest import RandomForestClassifier, RandomForestRegressor
from dtforest.data import make_blobs, make_moons, make_regression
from dtforest.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_squared_error, r2_score,
)
from dtforest.utils import train_test_split


def render_tree(node, depth=0, max_depth=3, prefix="", is_last=True, feature_names=None):
    """ASCII rendering of a small decision tree."""
    lines = []
    connector = "└── " if is_last else "├── "
    if node.is_leaf:
        val = node.value
        if isinstance(val, (int, np.integer)):
            extra = f"class={val}"
        else:
            extra = f"value={val:.3f}"
        lines.append(f"{prefix}{connector}[leaf] {extra} "
                     f"(n={node.n_samples}, impurity={node.impurity:.3f})")
    else:
        name = feature_names[node.feature_idx] if feature_names else f"x[{node.feature_idx}]"
        lines.append(f"{prefix}{connector}{name} <= {node.threshold:.3f}? "
                     f"(n={node.n_samples}, impurity={node.impurity:.3f})")
        if depth < max_depth:
            ext = "    " if is_last else "│   "
            lines += render_tree(node.left, depth + 1, max_depth, prefix + ext,
                                 False, feature_names)
            lines += render_tree(node.right, depth + 1, max_depth, prefix + ext,
                                 True, feature_names)
        else:
            ext = "    " if is_last else "│   "
            lines.append(f"{prefix}{ext}└── … (truncated)")
    return lines


def run_classify(args):
    X, y = make_moons(n_samples=args.n_samples, noise=0.15, random_state=args.seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=args.seed)

    t0 = time.perf_counter()
    tree = DecisionTreeClassifier(max_depth=args.depth, random_state=args.seed)
    tree.fit(Xtr, ytr)
    t_tree = time.perf_counter() - t0

    t0 = time.perf_counter()
    forest = RandomForestClassifier(n_estimators=args.trees, max_depth=args.depth,
                                    oob_score=True, random_state=args.seed)
    forest.fit(Xtr, ytr)
    t_forest = time.perf_counter() - t0

    tp, fp = tree.predict(Xte), forest.predict(Xte)
    print(f"dataset      : two moons, n={args.n_samples}, seed={args.seed}")
    print(f"single tree  : depth={args.depth}, test accuracy "
          f"{accuracy_score(yte, tp):.4f} ({t_tree:.2f}s)")
    print(f"forest       : {args.trees} trees, depth={args.depth}, test accuracy "
          f"{accuracy_score(yte, fp):.4f}, F1-macro {f1_score(yte, fp, average='macro'):.4f} "
          f"({t_forest:.2f}s)")
    print(f"OOB accuracy : {forest.oob_score_:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, fp))
    imp = forest.feature_importances_
    order = np.argsort(imp)[::-1]
    print("feature importances:")
    for j in order:
        print(f"  x[{j}]: {imp[j]:.4f} {'#' * int(40 * imp[j])}")
    if args.show_tree:
        if args.depth is not None and args.depth > 3:
            print("\n--show-tree renders only depth<=3; refit a shallow tree:")
        small = DecisionTreeClassifier(max_depth=min(args.depth or 3, 3),
                                       random_state=args.seed).fit(Xtr, ytr)
        print("\nsingle tree (depth<=3) structure:")
        print("\n".join(render_tree(small.tree_)))


def run_regress(args):
    X, y = make_regression(n_samples=args.n_samples, noise=0.5,
                           random_state=args.seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=args.seed)
    names = ["x0 (step)", "x1 (linear)", "x2 (quadratic)", "x3 (noise)"]

    t0 = time.perf_counter()
    tree = DecisionTreeRegressor(max_depth=args.depth, random_state=args.seed)
    tree.fit(Xtr, ytr)
    t_tree = time.perf_counter() - t0

    t0 = time.perf_counter()
    forest = RandomForestRegressor(n_estimators=args.trees, max_depth=args.depth,
                                   oob_score=True, random_state=args.seed)
    forest.fit(Xtr, ytr)
    t_forest = time.perf_counter() - t0

    tp, fp = tree.predict(Xte), forest.predict(Xte)
    print(f"dataset      : y = 10*[x0>0.5] + 3*x1 - 2*x2^2 + noise, "
          f"n={args.n_samples}, seed={args.seed}")
    print(f"single tree  : depth={args.depth}, test R² {r2_score(yte, tp):.4f}, "
          f"MSE {mean_squared_error(yte, tp):.4f} ({t_tree:.2f}s)")
    print(f"forest       : {args.trees} trees, depth={args.depth}, test R² "
          f"{r2_score(yte, fp):.4f}, MSE {mean_squared_error(yte, fp):.4f} "
          f"({t_forest:.2f}s)")
    print(f"OOB R²       : {forest.oob_score_:.4f}")
    imp = forest.feature_importances_
    order = np.argsort(imp)[::-1]
    print("feature importances:")
    for j in order:
        print(f"  {names[j]:<14}: {imp[j]:.4f} {'#' * int(40 * imp[j])}")
    if args.show_tree:
        small = DecisionTreeRegressor(max_depth=min(args.depth or 3, 3),
                                      random_state=args.seed).fit(Xtr, ytr)
        print("\nsingle tree (depth<=3) structure:")
        print("\n".join(render_tree(small.tree_, feature_names=names)))


def main():
    ap = argparse.ArgumentParser(description="Decision trees & random forests from scratch")
    ap.add_argument("--task", choices=["classify", "regress"], default="classify")
    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--trees", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show-tree", action="store_true",
                    help="render a small (depth<=3) single tree as ASCII art")
    args = ap.parse_args()
    if args.task == "classify":
        run_classify(args)
    else:
        run_regress(args)


if __name__ == "__main__":
    main()
