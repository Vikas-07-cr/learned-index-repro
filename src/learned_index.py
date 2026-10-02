"""Learned indexes (Kraska et al., 2018, "The Case for Learned Index Structures").

Key idea: a sorted array is a CDF. Position ~= F(key) * N. A model learns F,
predicts a position, and a tiny "last-mile" binary search inside a guaranteed
error window fixes the prediction. The window comes from the worst errors seen
on the training keys, so every stored key is ALWAYS found (correct by design).

Two versions:
  LinearLearnedIndex : one linear model for the whole key range.
  RMI                : Recursive Model Index, 2 stages. Stage 1 routes a key to one
                       of M small linear models (stage 2); each has its own window.
"""
import math
from bisect import bisect_left

import numpy as np


def _fit_line(x, y):
    """Least-squares line y = a*x + b (closed form, numerically safe on x in [0,1])."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()
    den = n * sxx - sx * sx
    if den <= 1e-18:
        return 0.0, float(sy / n)
    a = (n * sxy - sx * sy) / den
    return float(a), float((sy - a * sx) / n)


class LinearLearnedIndex:
    name = "LearnedLinear"

    def __init__(self, keys):
        self.n = len(keys)
        self.keys = keys.tolist()
        self.kmin = float(keys[0])
        self.span = float(keys[-1] - keys[0]) or 1.0
        x = (keys - self.kmin) / self.span
        pos = np.arange(self.n, dtype=np.float64)
        self.a, self.b = _fit_line(x, pos)
        pred = np.floor(self.a * x + self.b).astype(np.int64)
        err = np.arange(self.n) - pred
        self.lo_err = int(err.min())
        self.hi_err = int(err.max())
        self.abs_err = np.abs(err)

    def lookup(self, key):
        p = int(math.floor(self.a * ((key - self.kmin) / self.span) + self.b))
        lo = max(0, p + self.lo_err)
        hi = min(self.n, p + self.hi_err + 1)
        i = bisect_left(self.keys, key, lo, hi)
        return i if i < self.n and self.keys[i] == key else -1

    def size_bytes(self):
        return 5 * 8  # a, b, kmin, span + two error bounds (tiny!)

    def params(self):
        return "1 model"

    def error_stats(self):
        return float(self.abs_err.mean()), int(self.abs_err.max()), float(self.hi_err - self.lo_err + 1)


class RMI:
    name = "RMI"

    def __init__(self, keys, num_models=100):
        self.n = len(keys)
        self.m = int(num_models)
        self.keys = keys.tolist()
        self.kmin = float(keys[0])
        self.span = float(keys[-1] - keys[0]) or 1.0
        x = (keys - self.kmin) / self.span
        pos = np.arange(self.n, dtype=np.float64)

        # ---- stage 1: one line over everything, used only to ROUTE keys ----
        self.a1, self.b1 = _fit_line(x, pos)
        route = self._route_array(x)

        # ---- stage 2: one line per group of keys (closed-form, all groups at once) ----
        m = self.m
        cnt = np.bincount(route, minlength=m).astype(np.float64)
        sx = np.bincount(route, weights=x, minlength=m)
        sy = np.bincount(route, weights=pos, minlength=m)
        sxx = np.bincount(route, weights=x * x, minlength=m)
        sxy = np.bincount(route, weights=x * pos, minlength=m)
        den = cnt * sxx - sx * sx
        safe = np.where(den > 1e-18, den, 1.0)
        a2 = np.where(den > 1e-18, (cnt * sxy - sx * sy) / safe, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            b2 = np.where(cnt > 0, (sy - a2 * sx) / np.where(cnt > 0, cnt, 1.0), 0.0)
        self.a2, self.b2 = a2, b2

        # ---- per-model error bounds (the guarantee that makes it correct) ----
        pred = np.floor(a2[route] * x + b2[route]).astype(np.int64)
        err = np.arange(self.n) - pred
        lo = np.zeros(m, dtype=np.int64)
        hi = np.zeros(m, dtype=np.int64)
        np.minimum.at(lo, route, err)
        np.maximum.at(hi, route, err)
        self.lo_err, self.hi_err = lo, hi
        self.abs_err = np.abs(err)
        self.window = (hi - lo + 1)[cnt > 0]
        self._cnt = cnt

    # same arithmetic in vector and scalar form so routing always agrees
    def _route_array(self, x):
        p1 = np.floor(self.a1 * x + self.b1)
        return np.clip((p1 * self.m / self.n).astype(np.int64), 0, self.m - 1)

    def _route(self, x):
        p1 = math.floor(self.a1 * x + self.b1)
        j = int(p1 * self.m / self.n)
        return 0 if j < 0 else (self.m - 1 if j >= self.m else j)

    def lookup(self, key):
        x = (key - self.kmin) / self.span
        j = self._route(x)
        p = int(math.floor(self.a2[j] * x + self.b2[j]))
        lo = max(0, p + int(self.lo_err[j]))
        hi = min(self.n, p + int(self.hi_err[j]) + 1)
        i = bisect_left(self.keys, key, lo, hi)
        return i if i < self.n and self.keys[i] == key else -1

    def lookup_batch(self, queries):
        """Vectorised lookups (no Python loop): predict, then binary-search each window."""
        q = np.asarray(queries, dtype=np.float64)
        x = (q - self.kmin) / self.span
        j = self._route_array(x)
        p = np.floor(self.a2[j] * x + self.b2[j]).astype(np.int64)
        lo = np.maximum(0, p + self.lo_err[j])
        hi = np.minimum(self.n, p + self.hi_err[j] + 1)
        keys = self._np_keys()
        for _ in range(int(np.ceil(np.log2(max(2, (hi - lo).max() + 1)))) + 1):
            mid = (lo + hi) // 2
            go_right = keys[np.minimum(mid, self.n - 1)] < q
            go_right &= lo < hi
            lo = np.where(go_right, mid + 1, lo)
            hi = np.where(go_right | (lo >= hi), hi, mid)
        pos = np.minimum(lo, self.n - 1)
        return np.where(keys[pos] == q, pos, -1)

    def _np_keys(self):
        if not hasattr(self, "_nk"):
            self._nk = np.asarray(self.keys, dtype=np.float64)
        return self._nk

    def size_bytes(self):
        # stage-1 line (2 floats) + for each stage-2 model: slope, intercept (8 B each)
        # and two error bounds (4 B each) + a few header values
        return 2 * 8 + self.m * (8 + 8 + 4 + 4) + 3 * 8

    def params(self):
        return f"{self.m} models"

    def error_stats(self):
        """(mean abs error, max abs error, average search window in positions)."""
        w = float(np.average(self.window, weights=self._cnt[self._cnt > 0]))
        return float(self.abs_err.mean()), int(self.abs_err.max()), w
