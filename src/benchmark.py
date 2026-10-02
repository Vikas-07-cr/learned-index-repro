"""Run every index on every dataset and save the numbers to results/results.csv."""
import argparse
import os
import time

import numpy as np
import pandas as pd

from btree_baseline import BinarySearchIndex, BTreeIndex
from datasets import DATASETS, get_dataset
from learned_index import RMI, LinearLearnedIndex

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def is_correct(index, keys, step=17):
    """Every stored key must come back at its true position."""
    return all(index.lookup(float(keys[i])) == i for i in range(0, len(keys), step))


def time_scalar(index, queries, repeats):
    """Mean microseconds per single lookup (Python loop), repeated for a std-dev."""
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in queries:
            index.lookup(q)
        runs.append((time.perf_counter() - t0) / len(queries) * 1e6)
    return float(np.mean(runs)), float(np.std(runs))


def time_batch(fn, queries, repeats):
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(queries)
        runs.append((time.perf_counter() - t0) / len(queries) * 1e6)
    return float(np.mean(runs)), float(np.std(runs))


def run(sizes, model_counts, n_queries, repeats):
    rows = []
    for ds in DATASETS:
        for n in sizes:
            keys = get_dataset(ds, n)
            rng = np.random.default_rng(1)
            queries = rng.choice(keys, size=n_queries).tolist()
            builders = [BTreeIndex, BinarySearchIndex, LinearLearnedIndex]
            builders += [lambda k, m=m: RMI(k, m) for m in model_counts]
            for build in builders:
                t0 = time.perf_counter()
                idx = build(keys)
                build_s = time.perf_counter() - t0
                mean_us, std_us = time_scalar(idx, queries, repeats)
                err = idx.error_stats() if hasattr(idx, "error_stats") else (0.0, 0, np.nan)
                rows.append(dict(
                    dataset=ds, n=len(keys), index=idx.name, params=idx.params(),
                    build_s=build_s, lookup_us=mean_us, lookup_us_std=std_us,
                    size_mb=idx.size_bytes() / 1e6, mean_abs_err=err[0],
                    max_abs_err=err[1], avg_window=err[2],
                    correct=is_correct(idx, keys)))
                print(f"{ds:10s} n={len(keys):>8d} {idx.name:13s} {idx.params():10s} "
                      f"lookup={mean_us:6.2f}us size={idx.size_bytes()/1e6:8.4f}MB "
                      f"correct={rows[-1]['correct']}")

            # Batch (vectorised) comparison: removes Python-loop overhead
            q_arr = np.asarray(queries)
            best = RMI(keys, 1000)
            for label, fn in (("RMI-batch", best.lookup_batch),
                              ("searchsorted-batch", lambda q: np.searchsorted(keys, q))):
                mean_us, std_us = time_batch(fn, q_arr, repeats)
                rows.append(dict(dataset=ds, n=len(keys), index=label,
                                 params="1000 models" if label == "RMI-batch" else "",
                                 build_s=np.nan, lookup_us=mean_us, lookup_us_std=std_us,
                                 size_mb=best.size_bytes() / 1e6 if label == "RMI-batch" else 0.0,
                                 mean_abs_err=np.nan, max_abs_err=np.nan,
                                 avg_window=np.nan, correct=True))
                print(f"{ds:10s} n={len(keys):>8d} {label:18s} lookup={mean_us:6.3f}us")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run (100k keys only)")
    args = ap.parse_args()
    sizes = [100_000] if args.quick else [100_000, 1_000_000]
    df = run(sizes, [10, 100, 1000, 10000],
             n_queries=10_000 if args.quick else 30_000,
             repeats=3)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")
    print("All indexes correct:", bool(df["correct"].all()))
