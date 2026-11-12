"""Iris loader: downloads the CSV from UCI on first use and caches it.

The dataset is not bundled with this repository. The first run downloads
it (about 4 KB) directly from its owning source, the UCI Machine
Learning Repository, and caches it in data/, which is git-ignored.
Internet access is needed for that first download only.
"""

import os
import urllib.request

import numpy as np

URL = "https://archive.ics.uci.edu/static/public/53/data.csv"


def _ensure_cached(cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "iris.csv")
    if not os.path.exists(path):
        print("downloading Iris from UCI (one time only)...")
        urllib.request.urlretrieve(URL, path)
    return path


def load_iris(cache_dir="data"):
    """Return (X, y, class_names, feature_names) for the 150 Iris flowers."""
    path = _ensure_cached(cache_dir)
    lines = open(path).read().strip().splitlines()
    header, rows = lines[0].split(","), [l.split(",") for l in lines[1:]]
    feature_names = header[:4]
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    labels = np.array([r[4] for r in rows])
    class_names, y = np.unique(labels, return_inverse=True)
    return X, y, class_names, feature_names
