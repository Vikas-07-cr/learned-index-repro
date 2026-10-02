# Reproducing Learned Index Structures (B-Tree vs RMI)

Reproduction study of **"The Case for Learned Index Structures"** (Kraska, Beutel, Chi, Dean, Polyzotis; SIGMOD 2018, arXiv:1712.01208), written in Python.

**Question:** can a small machine-learning model replace a B-Tree for finding where a key sits in a sorted array, and what does it cost or save?

## The idea in 30 seconds
A sorted array of keys is just a cumulative distribution function (CDF): `position ≈ F(key) × N`.
A model learns `F`, predicts the position, and a tiny **last-mile binary search** inside a guaranteed
error window corrects the prediction. The window comes from the worst errors seen on the stored keys,
so every stored key is always found.

## What is implemented
| File | What it does |
|---|---|
| `src/datasets.py` | 3 synthetic key distributions: uniform, lognormal, clustered |
| `src/btree_baseline.py` | Baselines: real B-Tree (`BTrees` package) and plain binary search |
| `src/learned_index.py` | `LinearLearnedIndex` (one model) and `RMI` (2-stage Recursive Model Index) with per-model error bounds, plus a vectorised `lookup_batch` |
| `src/benchmark.py` | Runs everything, writes `results/results.csv` |
| `src/plots.py` | Makes `results/fig1..fig5*.png` |
| `tests/test_indexes.py` | Correctness tests (hits, misses, batch lookups) |
| `run_all.py` | One command to reproduce all results |

## How to run
```
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
pytest -q                      # 11 tests should pass
python run_all.py              # full run, about 15-60 seconds
python run_all.py --quick      # smaller test run
```
Results appear in `results/` (`results.csv` and the figures).

## What is measured
- Lookup time per query (Python loop) and **batch lookup time** (vectorised, no Python loop overhead)
- Index size (learned index: exact bytes of its parameters; B-Tree: standard size model, see limitations)
- Build time, prediction error, average last-mile search window
- Correctness of every index on every dataset

## Results
_Fill this section with YOUR numbers after running `python run_all.py`. Paste the figures from `results/` and write 3-5 sentences on each:_
1. Size: how much smaller is the RMI than the B-Tree?
2. Speed: scalar lookups and batch lookups. Is the learned index faster or slower here, and why?
3. Datasets: which distribution is hardest for the models? How does the number of models change the window?

## Limitations (be honest about these in the report)
- **Python overhead.** The B-Tree and `bisect` run in optimised C, while the learned index does its arithmetic in Python. Scalar timings therefore favour the baselines. The batch comparison removes the loop overhead and is the fairer speed comparison; a C/C++ implementation would be the real test (as in the paper).
- **B-Tree size is estimated** with a textbook model (16 B per entry, 69% page fill, fan-out 32) because the `BTrees` object cannot be pickled at this size without hitting Python's recursion limit.
- **Static data only.** Inserts and updates are not supported (the paper's basic version has the same limit).
- **Synthetic data** only. Real datasets (for example the SOSD benchmark) would strengthen the study.
- Linear models only at both stages; the paper also tries small neural networks at stage 1.

## Ideas for an extension
Insert support with a small delta buffer; neural network at stage 1; real-world datasets from SOSD; a Numba or C version of the lookup.

## Reference
Kraska et al., *The Case for Learned Index Structures*, SIGMOD 2018. https://arxiv.org/abs/1712.01208
