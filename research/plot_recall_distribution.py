import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"
img_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img"


def plot_recall_distribution(dataset: str, ef: int, algo: str = None):
    if algo:
        csv_dir = f"/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_{algo}"
    else:
        csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"
    path = os.path.join(csv_dir, f"per_query_baseline_{dataset}.csv")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    available_efs = sorted(df["EF"].unique())

    if ef not in available_efs:
        print(f"EF={ef} not found. Available: {available_efs}")
        return

    recalls = df[df["EF"] == ef]["Recall"].values
    avg_recall = recalls.mean()

    fig, ax = plt.subplots(figsize=(12, 7))

    n, bins, patches = ax.hist(
        recalls, bins=30, color="lightblue", edgecolor="black", alpha=0.85
    )

    kde = gaussian_kde(recalls, bw_method=0.15)
    x_range = np.linspace(recalls.min(), recalls.max(), 500)
    kde_values = kde(x_range)
    scale = n.max() / kde_values.max()
    ax.plot(x_range, kde_values * scale, color="steelblue", linewidth=2)

    ax.axvline(
        avg_recall,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Average Recall: {avg_recall:.4f}",
    )

    ax.set_title(
        f"Recall Distribution of Queries\\nDataset: {dataset} | ef={ef}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Number of Queries", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    os.makedirs(img_dir, exist_ok=True)
    out_path = os.path.join(img_dir, f"recall_dist_{dataset}_ef{ef}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g. deep-image-96-angular)"
    )
    parser.add_argument("--ef", required=True, type=int, help="EF value to plot")
    parser.add_argument("--algo", type=str, choices=["shiro", "ada"], help="Algorithm dir to read from")
    args = parser.parse_args()
    plot_recall_distribution(args.dataset, args.ef, args.algo)
