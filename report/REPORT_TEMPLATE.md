# Reproducing "The Case for Learned Index Structures"
Vikas · B.Tech CSE (3rd year), SRM Ramapuram · [date]

## 1. Introduction (half a page)
- What problem do indexes solve (find a key's position in a sorted array)?
- The paper's claim in one paragraph (a model that learns the CDF can replace a B-Tree).
- What this project does: reproduce the core idea in Python and test it on 3 key distributions.

## 2. Method (1 page)
- B-Tree and binary-search baselines.
- Single linear model + error bounds + last-mile search.
- 2-stage RMI: stage 1 routes, stage 2 has M linear models, per-model error bounds.
- Why every stored key is always found (the error-bound argument).
- Datasets: uniform, lognormal, clustered. Sizes: 100k and 1M keys.
- Measurements: lookup latency (scalar and batch), size, build time, error, window.

## 3. Results (1 page)
Insert fig1 to fig5 from `results/` and explain each in 3-5 sentences.

## 4. Discussion (half a page)
- Did you reproduce the main claim? Which parts yes, which parts no?
- Why did the learned index struggle on some distributions?
- Effect of the number of stage-2 models (size vs accuracy).

## 5. Limitations and future work (half a page)
- Python overhead, estimated B-Tree size, static data, synthetic data.
- Next steps: inserts with a delta buffer, neural network at stage 1, real SOSD datasets, C/Numba implementation.

## References
Kraska et al., SIGMOD 2018, arXiv:1712.01208.
