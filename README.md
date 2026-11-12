# Iris Decision Tree (NumPy from scratch)

A small decision tree classifier (CART-style, Gini splits) built entirely
with NumPy: no scikit-learn, no framework. Split search, impurity and the
recursion are implemented by hand so every step is readable, and the
fitted tree prints as plain rules you can inspect.

Built by Yashraj Pangave.

## What it does

- Trains a binary decision tree on the classic Iris flowers data (150
  flowers, 4 measurements each, 3 species) using a seeded stratified
  split: 120 training and 30 held-out test flowers.
- With the default settings (`python src/train.py`, max depth 5, split
  seed 42) the measured result is 93.3% accuracy on the 30 test flowers
  (28 of 30). The run is deterministic, so the same command reproduces
  the same number and the same tree.
- Depth matters more than you might think on this tiny dataset, measured
  on the same split:

  | max depth | test accuracy |
  |-----------|---------------|
  | 2         | 93.3% (28/30) |
  | 3         | 96.7% (29/30) |
  | 5         | 93.3% (28/30) |

- Both misses at depth 5 are the classic hard pair: Iris-virginica
  flowers read as Iris-versicolor, whose measurements genuinely overlap.

## Quick start

```bash
pip install -r requirements.txt   # numpy only
python src/train.py               # downloads the data once, trains, prints the rules
python src/train.py --max-depth 3 # the best depth measured on this split
python tests/test_tree.py         # sanity tests
```

The dataset is not bundled with this repository. The first run downloads
it (about 4 KB) directly from its owning source, the UCI Machine
Learning Repository, and caches it in `data/`, which is git-ignored.
Internet access is needed for that first download only.

Example output tree (depth 5): the first question is always
`petal length <= 2.6`, which alone sorts out every setosa.

## Data source and terms

Iris - R. A. Fisher (1936), "The use of multiple measurements in
taxonomic problems", via the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/53/iris

This project does not redistribute the dataset: the code downloads it
directly from UCI on your machine. The MIT license in this repository
covers the code only, not the data. If you use the data, cite Fisher
(1936) and respect any terms UCI sets.

## Project layout

```
src/tree.py    the tree: Gini splits, recursion, readable-rule printer
src/data.py    download-on-first-use loader for the UCI CSV
src/train.py   stratified split, training, measured accuracy, tree printout
tests/         sanity tests (data shape, Gini values, split seeding,
               learnability, determinism)
data/          local dataset cache (git-ignored, created on first run)
```

## Why from scratch?

A decision tree is small enough to hold in your head end to end: sweep
every possible split, keep the one that best purifies the classes,
recurse. Writing that loop by hand makes bias and variance concrete -
the depth table above is overfitting and underfitting you can see.
