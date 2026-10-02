import numpy as np


def uniform(n, seed=0):
    """Keys spread evenly. Easiest case for a learned index."""
    rng = np.random.default_rng(seed)
    return np.unique(rng.integers(0, 10**12, size=n)).astype(np.float64)


def lognormal(n, seed=0):
    """Keys bunched near small values with a long tail. Harder case."""
    rng = np.random.default_rng(seed)
    return np.unique(rng.lognormal(0, 2, size=n) * 1e9)


def clustered(n, seed=0):
    """Keys grouped into 20 tight clusters with big gaps. Hardest case."""
    rng = np.random.default_rng(seed)
    centers = rng.integers(0, 10**12, size=20)
    pts = np.concatenate([rng.normal(c, 1e6, n // 20) for c in centers])
    return np.unique(pts)


def get_dataset(name, n, seed=0):
    makers = {"uniform": uniform, "lognormal": lognormal, "clustered": clustered}
    return makers[name](n, seed)


if __name__ == "__main__":
    for name in ["uniform", "lognormal", "clustered"]:
        keys = get_dataset(name, 100_000)
        is_sorted = bool(np.all(keys[:-1] < keys[1:]))
        print(f"{name}: {len(keys)} keys, sorted and unique = {is_sorted}")