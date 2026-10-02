"""Draw ONE picture that explains the whole idea: results/fig0_how_it_works.png

Left : a sorted key list is a curve (position vs key). A model is a line through it.
Right: zoom in - the model's guess is slightly off, and the error window around the
       guess is where the final tiny binary search happens.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets import get_dataset
from learned_index import RMI, LinearLearnedIndex

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "fig0_how_it_works.png")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    n = 20_000
    for ax, name in zip(axes[:2], ["uniform", "clustered"]):
        keys = get_dataset(name, n)
        pos = np.arange(len(keys))
        lin = LinearLearnedIndex(keys)
        x = (keys - lin.kmin) / lin.span
        ax.plot(keys, pos, lw=2, label="true position of each key")
        ax.plot(keys, lin.a * x + lin.b, "r--", lw=1.5, label="one linear model")
        ax.set_title(f"{name} keys: the curve vs one line")
        ax.set_xlabel("key")
        ax.set_ylabel("position in sorted list")
        ax.legend(fontsize=8)

    ax = axes[2]
    keys = get_dataset("uniform", n)
    rmi = RMI(keys, 200)
    i0 = 9_000
    idx = np.arange(i0, i0 + 120)
    k = keys[idx]
    x = (k - rmi.kmin) / rmi.span
    j = rmi._route_array(x)
    pred = np.floor(rmi.a2[j] * x + rmi.b2[j])
    ax.plot(k, idx, "b.", label="true position")
    ax.plot(k, pred, "r-", lw=1.5, label="model guess")
    ax.fill_between(k, pred + rmi.lo_err[j], pred + rmi.hi_err[j], color="orange",
                    alpha=0.35, label="error window = last-mile search")
    ax.set_title("Zoom: guess + window always contains the truth")
    ax.set_xlabel("key")
    ax.set_ylabel("position")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("saved", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
