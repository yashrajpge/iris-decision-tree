"""Train the decision tree on Iris with a seeded stratified split.

Usage: python src/train.py [--max-depth 5] [--seed 42]
Prints the measured accuracy on the held-out flowers and the tree rules.
"""

import argparse

import numpy as np

from data import load_iris
from tree import DecisionTree


def stratified_split(y, test_fraction=0.2, seed=42):
    """Seeded split keeping each class proportion in both parts."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * test_fraction))
        test_idx.extend(idx[:k])
    test_idx = np.array(sorted(test_idx))
    mask = np.ones(len(y), dtype=bool)
    mask[test_idx] = False
    return np.where(mask)[0], test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, class_names, feature_names = load_iris()
    train_idx, test_idx = stratified_split(y, seed=args.seed)
    tree = DecisionTree(max_depth=args.max_depth).fit(X[train_idx], y[train_idx])

    pred = tree.predict(X[test_idx])
    correct = int((pred == y[test_idx]).sum())
    total = len(test_idx)
    print(f"max_depth {args.max_depth}, split seed {args.seed}")
    print(f"train {len(train_idx)} flowers, test {total} flowers (stratified)")
    print(f"test accuracy: {correct / total:.4f} ({correct}/{total})")
    print(f"tree depth {tree.depth()}, {tree.n_leaves()} leaves")
    print()
    for line in tree.describe(feature_names):
        print(line)
    errors = np.where(pred != y[test_idx])[0]
    for i in errors:
        true_name = class_names[y[test_idx][i]]
        print(f"missed: test flower {i} is {true_name}, read as {class_names[pred[i]]}")


if __name__ == "__main__":
    main()
