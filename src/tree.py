"""A CART-style decision tree classifier written from scratch with NumPy.

Each split is chosen by scanning every feature threshold and picking the
one with the best Gini gain; recursion stops at max_depth,
min_samples_split, a pure node, or when no split helps. The fitted tree
can be printed as readable rules.
"""

import numpy as np


def gini(counts):
    """Gini impurity of a class-count vector: 1 - sum(p^2)."""
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return float(1.0 - (p ** 2).sum())


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "counts", "label")

    def __init__(self, counts, label):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.counts = counts
        self.label = label


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, yint = np.unique(y, return_inverse=True)
        self.root_ = self._grow(X, yint, depth=0)
        return self

    def _best_split(self, X, y):
        """Sweep every feature threshold; return (gain, feature, threshold)."""
        parent_counts = np.bincount(y, minlength=len(self.classes_))
        parent_imp = gini(parent_counts)
        n = len(y)
        best = None
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f], kind="mergesort")
            xs, ys = X[order, f], y[order]
            left_counts = np.zeros(len(self.classes_), dtype=int)
            for i in range(1, n):
                left_counts[ys[i - 1]] += 1
                if xs[i] == xs[i - 1]:
                    continue
                right_counts = parent_counts - left_counts
                imp = (i * gini(left_counts) + (n - i) * gini(right_counts)) / n
                gain = parent_imp - imp
                if best is None or gain > best[0] + 1e-12:
                    best = (gain, f, float(xs[i - 1] + xs[i]) / 2.0)
        return best

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        node = _Node(counts, int(counts.argmax()))
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or gini(counts) == 0.0
        ):
            return node
        split = self._best_split(X, y)
        if split is None or split[0] <= 1e-12:
            return node
        _, f, t = split
        left = X[:, f] <= t
        if left.all() or (~left).all():
            return node
        node.feature, node.threshold = f, t
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], depth + 1)
        return node

    def _predict_one(self, x, node):
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.label

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.classes_[[self._predict_one(x, self.root_) for x in X]]

    def depth(self):
        def d(node):
            if node.feature is None:
                return 0
            return 1 + max(d(node.left), d(node.right))

        return d(self.root_)

    def n_leaves(self):
        def count(node):
            if node.feature is None:
                return 1
            return count(node.left) + count(node.right)

        return count(self.root_)

    def describe(self, feature_names=None):
        """The tree as readable rules, one line per node."""
        lines = []

        def walk(node, indent):
            pad = "  " * indent
            if node.feature is None:
                name = self.classes_[node.label]
                lines.append(f"{pad}-> {name} (samples: {node.counts.sum()})")
                return
            fname = (
                feature_names[node.feature]
                if feature_names is not None
                else f"x[{node.feature}]"
            )
            lines.append(f"{pad}if {fname} <= {node.threshold:.3f}:")
            walk(node.left, indent + 1)
            lines.append(f"{pad}else:")
            walk(node.right, indent + 1)

        walk(self.root_, 0)
        return lines
