"""See a learned index working, step by step.   python demo.py

It builds an index on 1,000,000 keys, then looks up one key and explains every step.
Try:  python demo.py uniform | lognormal | clustered       (default: uniform)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from bisect import bisect_left  # noqa: E402

import numpy as np  # noqa: E402

from btree_baseline import BTreeIndex  # noqa: E402
from datasets import DATASETS, get_dataset  # noqa: E402
from learned_index import RMI  # noqa: E402


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "uniform"
    if name not in DATASETS:
        sys.exit(f"Choose one of: {', '.join(DATASETS)}")
    n, m = 1_000_000, 1000

    print(f"\n=== Learned index demo ({name} keys) ===\n")
    keys = get_dataset(name, n)
    print(f"1) Made {len(keys):,} sorted unique keys.  First 3: {keys[:3]}")

    t0 = time.perf_counter()
    rmi = RMI(keys, m)
    print(f"2) Trained an RMI with {m} small models in {time.perf_counter()-t0:.2f}s.")
    print(f"   Its whole size: {rmi.size_bytes()/1024:.1f} KB")
    bt = BTreeIndex(keys)
    print(f"   A B-Tree on the same keys (estimated): {bt.size_bytes()/1e6:.1f} MB "
          f"-> about {bt.size_bytes()/rmi.size_bytes():,.0f}x bigger\n")

    true_pos = 654_321
    key = float(keys[true_pos])
    print(f"3) Look up the key {key:,.2f}  (really at position {true_pos:,})")
    x = (key - rmi.kmin) / rmi.span
    j = rmi._route(x)
    print(f"   Stage 1 says: 'use model number {j}'")
    guess = int(math.floor(rmi.a2[j] * x + rmi.b2[j]))
    print(f"   Model {j} guesses position {guess:,}   (off by {true_pos-guess:+,})")
    lo = max(0, guess + int(rmi.lo_err[j]))
    hi = min(len(keys), guess + int(rmi.hi_err[j]) + 1)
    print(f"   Its worst-case error is known, so the answer must be in [{lo:,}, {hi:,})")
    print(f"   That is only {hi-lo:,} positions out of {len(keys):,} to search")
    found = bisect_left(rmi.keys, key, lo, hi)
    print(f"   Binary search inside that window finds position {found:,}  "
          f"-> {'CORRECT' if found == true_pos else 'WRONG'}")
    steps_full = math.ceil(math.log2(len(keys)))
    steps_win = max(1, math.ceil(math.log2(hi - lo)))
    print(f"   Search steps: about {steps_win} (window) instead of {steps_full} (whole list)\n")

    q = np.random.default_rng(0).choice(keys, 2000)
    ok = all(rmi.lookup(float(v)) == int(np.searchsorted(keys, v)) for v in q)
    print(f"4) Checked 2,000 random lookups against numpy: all correct = {ok}")
    e = rmi.error_stats()
    print(f"5) Average error window on this data: {e[2]:,.0f} positions "
          f"(smaller = better model)\n")


if __name__ == "__main__":
    main()
