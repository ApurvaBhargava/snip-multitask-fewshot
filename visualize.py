import json
import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Load + improve DataFrame
# ---------------------------------------------------------------------
def compute_snip_improvement(results):
    rows = []

    for k in sorted(results.keys(), key=lambda x: int(x)):
        dense_acc = results[k]["dense"]

        for mode in ["labeled", "unlabeled", "crossdomain"]:
            for sparsity, snip_acc in results[k]["snip"][mode].items():

                pct_improve = (
                    ((snip_acc - dense_acc) / dense_acc) * 100.0
                    if dense_acc != 0 else float("nan")
                )

                rows.append({
                    "k_shot": int(k),
                    "mode": mode,
                    "sparsity": float(sparsity),
                    "snip_acc": snip_acc,
                    "dense_acc": dense_acc,
                    "absolute_gain": snip_acc - dense_acc,
                    "pct_improvement": pct_improve
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def ensure_dir(path):
    path = Path(path)
    path.mkdir(exist_ok=True)
    return path


# ---------------------------------------------------------------------
# PLOT SUITE BELOW
# ---------------------------------------------------------------------

# 1. Combined Accuracy vs Sparsity (per k)
def plot_combined_acc_vs_sparsity(results, outdir):
    sns.set(style="whitegrid", font_scale=1.2)
    sparsities = [0.5, 0.7, 0.9, 0.95]

    for k in sorted(results.keys(), key=lambda x: int(x)):
        dense_acc = results[k]["dense"]

        snip_labeled = [results[k]["snip"]["labeled"][str(s)] for s in sparsities]
        snip_unlabeled = [results[k]["snip"]["unlabeled"][str(s)] for s in sparsities]
        snip_cross = [results[k]["snip"]["crossdomain"][str(s)] for s in sparsities]

        plt.figure(figsize=(9, 6))
        plt.plot(sparsities, [dense_acc]*4, "--o", label="Dense Baseline")
        plt.plot(sparsities, snip_labeled, "-s", label="SNIP Labeled")
        plt.plot(sparsities, snip_unlabeled, "-^", label="SNIP Unlabeled")
        plt.plot(sparsities, snip_cross, "-x", label="SNIP CrossDomain")

        plt.xlabel("Sparsity")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy vs Sparsity (k={k})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"combined_acc_vs_sparsity_k{k}.png", dpi=300)
        plt.close()


# 2. Accuracy vs K-shot for each sparsity
def plot_acc_vs_k(results, outdir):
    sns.set(style="whitegrid", font_scale=1.2)
    sparsities = [0.5, 0.7, 0.9, 0.95]
    k_values = sorted([int(k) for k in results.keys()])

    for sp in sparsities:
        dense_acc = [results[str(k)]["dense"] for k in k_values]
        snip_labeled = [results[str(k)]["snip"]["labeled"][str(sp)] for k in k_values]
        snip_unlabeled = [results[str(k)]["snip"]["unlabeled"][str(sp)] for k in k_values]
        snip_cross = [results[str(k)]["snip"]["crossdomain"][str(sp)] for k in k_values]

        plt.figure(figsize=(9, 6))
        plt.plot(k_values, dense_acc, "-o", label="Dense")
        plt.plot(k_values, snip_labeled, "-s", label=f"Labeled (sp={sp})")
        plt.plot(k_values, snip_unlabeled, "-^", label=f"Unlabeled (sp={sp})")
        plt.plot(k_values, snip_cross, "-x", label=f"CrossDomain (sp={sp})")

        plt.xlabel("K-Shot")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy vs K-shot (Sparsity={sp})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"acc_vs_k_sp{sp}.png", dpi=300)
        plt.close()


# 3. Accuracy vs Sparsity for each SNIP mode (per k)
def plot_acc_vs_sparsity_for_each_mode(results, outdir, k=5):
    sns.set(style="whitegrid", font_scale=1.2)
    sparsities = [0.5, 0.7, 0.9, 0.95]

    labeled = [results[str(k)]["snip"]["labeled"][str(s)] for s in sparsities]
    unlabeled = [results[str(k)]["snip"]["unlabeled"][str(s)] for s in sparsities]
    cross = [results[str(k)]["snip"]["crossdomain"][str(s)] for s in sparsities]

    plt.figure(figsize=(9, 6))
    plt.plot(sparsities, labeled, "-s", label="Labeled")
    plt.plot(sparsities, unlabeled, "-^", label="Unlabeled")
    plt.plot(sparsities, cross, "-x", label="CrossDomain")

    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Sparsity (k={k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"acc_vs_sparsity_k{k}.png", dpi=300)
    plt.close()


# 4. Heatmaps (3 modes × sparsity × k)
def plot_heatmaps(results, outdir):
    sns.set(font_scale=1.2)
    modes = ["labeled", "unlabeled", "crossdomain"]
    sparsities = [0.5, 0.7, 0.9, 0.95]
    k_values = sorted([int(k) for k in results.keys()])

    for mode in modes:
        data = [[results[str(k)]["snip"][mode][str(sp)] for sp in sparsities]
                for k in k_values]

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            data,
            annot=True, fmt=".2f",
            xticklabels=sparsities,
            yticklabels=k_values,
            cmap="viridis"
        )
        plt.xlabel("Sparsity")
        plt.ylabel("K-shot")
        plt.title(f"Heatmap: {mode}")
        plt.tight_layout()
        plt.savefig(outdir / f"heatmap_{mode}.png", dpi=300)
        plt.close()


# 5. Multi-line grid (3-subplots)
def plot_mode_grid(results, outdir):
    sns.set(style="whitegrid", font_scale=1.1)
    sparsities = [0.5, 0.7, 0.9, 0.95]
    k_values = sorted([int(k) for k in results.keys()])
    modes = ["labeled", "unlabeled", "crossdomain"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        for sp in sparsities:
            y = [results[str(k)]["snip"][mode][str(sp)] for k in k_values]
            ax.plot(k_values, y, marker="o", label=f"sp={sp}")

        ax.set_title(mode)
        ax.set_xlabel("K-shot")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(outdir / "snip_mode_grid.png", dpi=300)
    plt.close()


# 6. Improvement heatmaps
def plot_improvement_heatmaps(df, outdir):
    pivot_abs = df.pivot_table(
        index="k_shot",
        columns=["mode", "sparsity"],
        values="absolute_gain"
    )
    pivot_pct = df.pivot_table(
        index="k_shot",
        columns=["mode", "sparsity"],
        values="pct_improvement"
    )

    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot_abs, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Absolute Accuracy Gain (SNIP - Dense)")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_absolute_gain.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot_pct, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("% Improvement Over Dense")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_pct_gain.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results.json")
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)

    # Load results
    with open(args.results, "r") as f:
        results = json.load(f)

    # Compute improvements
    df = compute_snip_improvement(results)
    df.to_csv(outdir / "snip_improvements.csv", index=False)

    # Full visualization suite
    plot_combined_acc_vs_sparsity(results, outdir)
    plot_acc_vs_k(results, outdir)
    plot_acc_vs_sparsity_for_each_mode(results, outdir, k=5)
    plot_heatmaps(results, outdir)
    plot_mode_grid(results, outdir)
    plot_improvement_heatmaps(df, outdir)

    print(f"\nAll plots & CSV saved in: {outdir.resolve()}\n")


if __name__ == "__main__":
    main()