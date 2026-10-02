"""Correctness tests. Run with:  pytest -q
A fast index that returns wrong positions is useless, so we test this first."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from btree_baseline import BinarySearchIndex, BTreeIndex  # noqa: E402
from datasets import DATASETS, get_dataset  # noqa: E402
from learned_index import RMI, LinearLearnedIndex  # noqa: E402


def all_indexes(keys):
    return [BTreeIndex(keys), BinarySearchIndex(keys), LinearLearnedIndex(keys),
            RMI(keys, 10), RMI(keys, 1000)]


@pytest.mark.parametrize("name", list(DATASETS))
def test_datasets_sorted_unique(name):
    keys = get_dataset(name, 20_000)
    assert np.all(keys[:-1] < keys[1:])


@pytest.mark.parametrize("name", list(DATASETS))
def test_every_key_found_at_true_position(name):
    keys = get_dataset(name, 20_000)
    for idx in all_indexes(keys):
        for i in range(0, len(keys), 7):
            assert idx.lookup(float(keys[i])) == i, (idx.name, i)


@pytest.mark.parametrize("name", list(DATASETS))
def test_missing_keys_return_minus_one(name):
    keys = get_dataset(name, 20_000)
    missing = [keys[0] - 1.0, keys[-1] + 1.0, (keys[10] + keys[11]) / 2]
    for idx in all_indexes(keys):
        for m in missing:
            assert idx.lookup(float(m)) == -1, (idx.name, m)


def test_rmi_batch_matches_numpy():
    keys = get_dataset("lognormal", 50_000)
    rmi = RMI(keys, 500)
    q = np.random.default_rng(3).choice(keys, 2000)
    assert np.array_equal(rmi.lookup_batch(q), np.searchsorted(keys, q))


def test_more_models_never_make_window_bigger_on_uniform():
    keys = get_dataset("uniform", 50_000)
    w10 = RMI(keys, 10).error_stats()[2]
    w1000 = RMI(keys, 1000).error_stats()[2]
    assert w1000 < w10
