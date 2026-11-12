"""Sanity tests: run with `python tests/test_tree.py` from the repo root.

The first run downloads the dataset from UCI (needs internet once).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from data import load_iris
from tree import DecisionTree, gini
from train import stratified_split


def test_iris_shape():
    X, y, class_names, feature_names = load_iris()
    assert X.shape == (150, 4), X.shape
    assert len(class_names) == 3
    assert (np.bincount(y) == 50).all()
    assert 0.0 < X.min() and X.max() < 10.0
    print("ok: 150 flowers, 4 features, 3 balanced classes")


def test_gini_values():
    assert gini(np.array([0, 5])) == 0.0
    assert gini(np.array([1, 1])) == 0.5
    assert abs(gini(np.array([1, 1, 1, 1])) - 0.75) < 1e-12
    print("ok: gini impurity values")


def test_tree_learns_simple_data():
    X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTree().fit(X, y)
    assert (tree.predict(X) == y).all()
    assert tree.depth() == 1
    print("ok: perfectly separable data is learned with one split")


def test_split_is_stratified_and_seeded():
    X, y, _, _ = load_iris()
    train_idx, test_idx = stratified_split(y, seed=42)
    assert len(train_idx) == 120 and len(test_idx) == 30
    assert (np.bincount(y[test_idx]) == 10).all()
    train2, test2 = stratified_split(y, seed=42)
    assert (test_idx == test2).all()
    print("ok: 120/30 split keeps 10 of each class in test and repeats with the seed")


def test_tree_is_deterministic():
    X, y, _, feature_names = load_iris()
    train_idx, _ = stratified_split(y, seed=42)
    a = DecisionTree(max_depth=5).fit(X[train_idx], y[train_idx])
    b = DecisionTree(max_depth=5).fit(X[train_idx], y[train_idx])
    assert (a.predict(X) == b.predict(X)).all()
    assert a.describe(feature_names) == b.describe(feature_names)
    print("ok: same data and settings grow the identical tree")


if __name__ == "__main__":
    test_iris_shape()
    test_gini_values()
    test_tree_learns_simple_data()
    test_split_is_stratified_and_seeded()
    test_tree_is_deterministic()
    print("all tests passed")
