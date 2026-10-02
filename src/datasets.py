"""Synthetic key sets used to test the indexes.

An index needs a SORTED list of UNIQUE numbers (keys). The shape of the
key distribution decides how easy it is for a model to learn it:
  - uniform   : evenly spread        -> easiest
  - lognormal : bunched, long tail   -> harder
  - clustered : 20 groups, big gaps  -> hardest
"""
import numpy as np


def uniform(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.unique(rng.integers(0, 10**12, size=n)).astype(np.float64)


def lognormal(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.unique(rng.lognormal(0, 2, size=n) * 1e9)


def clustered(n, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.integers(0, 10**12, size=20)
    pts = np.concatenate([rng.normal(c, 1e6, n // 20) for c in centers])
    return np.unique(pts)


DATASETS = {"uniform": uniform, "lognormal": lognormal, "clustered": clustered}


def get_dataset(name, n, seed=0):
    return DATASETS[name](n, seed)


if __name__ == "__main__":
    for name in DATASETS:
        keys = get_dataset(name, 100_000)
        ok = bool(np.all(keys[:-1] < keys[1:]))
        print(f"{name}: {len(keys)} keys, sorted and unique = {ok}")
