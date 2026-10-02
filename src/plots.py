"""Turn results/results.csv into the figures used in the report."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATASET_ORDER = ["uniform", "lognormal", "clustered"]


def load():
    return pd.read_csv(os.path.join(RESULTS, "results.csv"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, name), dpi=150)
    plt.close(fig)
    print("saved", name)


def latency_by_dataset(df):
    n = df["n"].max()
    d = df[df["n"] <= n]
    d = d[(d["n"] >= n * 0.9)]
    series = [("BTree", ""), ("BinarySearch", ""), ("LearnedLinear", "1 model"),
              ("RMI", "1000 models")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.2
    for i, (idx, params) in enumerate(series):
        vals = []
        for ds in DATASET_ORDER:
            r = d[(d["dataset"] == ds) & (d["index"] == idx) & (d["params"].fillna("") == params)]
            vals.append(r["lookup_us"].iloc[0] if len(r) else 0)
        ax.bar([x + i * w for x in range(3)], vals, w, label=f"{idx} {params}".strip())
    ax.set_xticks([x + 1.5 * w for x in range(3)])
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Mean lookup time (microseconds, Python)")
    ax.set_title(f"Lookup latency per dataset (n ~ {n:,})")
    ax.legend()
    save(fig, "fig1_latency_by_dataset.png")


def size_vs_models(df):
    n = df["n"].max()
    d = df[df["n"] >= n * 0.9]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    r = d[(d["index"] == "RMI") & (d["dataset"] == "uniform")].copy()
    r["m"] = r["params"].str.split().str[0].astype(int)
    r = r.sort_values("m")
    ax.plot(r["m"], r["size_mb"] * 1024, "o-", label="RMI")
    bt = d[(d["index"] == "BTree") & (d["dataset"] == "uniform")]["size_mb"].iloc[0]
    ax.axhline(bt * 1024, color="red", ls="--", label="B-Tree (estimated)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of stage-2 models")
    ax.set_ylabel("Index size (KB)")
    ax.set_title("Index size: RMI vs B-Tree")
    ax.legend()
    save(fig, "fig2_size_vs_models.png")


def error_vs_models(df):
    n = df["n"].max()
    d = df[(df["n"] >= n * 0.9) & (df["index"] == "RMI")].copy()
    d["m"] = d["params"].str.split().str[0].astype(int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ds in DATASET_ORDER:
        r = d[d["dataset"] == ds].sort_values("m")
        ax.plot(r["m"], r["avg_window"], "o-", label=ds)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of stage-2 models")
    ax.set_ylabel("Average last-mile search window (positions)")
    ax.set_title("Prediction quality: more models -> smaller window")
    ax.legend()
    save(fig, "fig3_window_vs_models.png")


def tradeoff(df):
    n = df["n"].max()
    d = df[(df["n"] >= n * 0.9) & (~df["index"].str.contains("batch"))]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    marker = {"uniform": "o", "lognormal": "s", "clustered": "^"}
    color = {"BTree": "tab:red", "BinarySearch": "tab:gray",
             "LearnedLinear": "tab:green", "RMI": "tab:blue"}
    for _, r in d.iterrows():
        ax.scatter(max(r["size_mb"], 1e-5), r["lookup_us"], c=color[r["index"]],
                   marker=marker[r["dataset"]], alpha=0.8)
    for k, c in color.items():
        ax.scatter([], [], c=c, label=k)
    for k, m in marker.items():
        ax.scatter([], [], c="black", marker=m, label=k)
    ax.set_xscale("log")
    ax.set_xlabel("Index size (MB, log)")
    ax.set_ylabel("Mean lookup time (us)")
    ax.set_title("Size vs speed trade-off")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "fig4_tradeoff.png")


def batch_comparison(df):
    n = df["n"].max()
    d = df[(df["n"] >= n * 0.9) & (df["index"].str.contains("batch"))]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    w = 0.35
    for i, idx in enumerate(["RMI-batch", "searchsorted-batch"]):
        vals = [d[(d["dataset"] == ds) & (d["index"] == idx)]["lookup_us"].iloc[0]
                for ds in DATASET_ORDER]
        ax.bar([x + i * w for x in range(3)], vals, w, label=idx)
    ax.set_xticks([x + w / 2 for x in range(3)])
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Microseconds per lookup (vectorised)")
    ax.set_title("Batch lookups (no Python loop overhead)")
    ax.legend()
    save(fig, "fig5_batch_comparison.png")


if __name__ == "__main__":
    data = load()
    latency_by_dataset(data)
    size_vs_models(data)
    error_vs_models(data)
    tradeoff(data)
    batch_comparison(data)
