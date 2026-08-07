import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_annotated_plot_with_stats(dataset_name: str, algo: str):
    print(f"Loading datasets for {dataset_name} ({algo})...")
    csv_dir = f"/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_{algo}"
    df_mine = pd.read_csv(
        f"{csv_dir}/per_query_results_{dataset_name}.csv"
    )
    df_base = pd.read_csv(
        f"{csv_dir}/per_query_baseline_{dataset_name}.csv"
    )

    avg_recall_mine = df_mine["Recall"].mean()
    best_ef = df_base["EF"].unique()[0]
    min_diff = 1.0
    for ef in df_base["EF"].unique():
        r = df_base[df_base["EF"] == ef]["Recall"].mean()
        if abs(r - avg_recall_mine) < min_diff:
            min_diff = abs(r - avg_recall_mine)
            best_ef = ef

    df_base_best = df_base[df_base["EF"] == best_ef].copy()
    df_merged = pd.merge(
        df_base_best, df_mine, on="QueryID", suffixes=("_Base", "_Mine")
    )

    q99_mine = df_merged["Latency(ns)_Mine"].quantile(0.99)
    q99_base = df_merged["Latency(ns)_Base"].quantile(0.99)
    df_merged = df_merged[
        (df_merged["Latency(ns)_Mine"] <= q99_mine)
        & (df_merged["Latency(ns)_Base"] <= q99_base)
    ]

    # Calculate stats BEFORE smoothing
    avg_rec_base = df_merged["Recall_Base"].mean()
    med_lat_base = df_merged["Latency(ns)_Base"].median() / 1e6
    avg_rec_mine = df_merged["Recall_Mine"].mean()
    med_lat_mine = df_merged["Latency(ns)_Mine"].median() / 1e6

    # Sort
    df_merged.sort_values(
        ["Recall_Base", "Latency(ns)_Base"], ascending=[True, False], inplace=True
    )
    df_merged.reset_index(drop=True, inplace=True)
    percentiles = np.linspace(0, 100, len(df_merged))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    window = max(1, len(df_merged) // 30)

    # min_periods=1 ensures the line goes all the way to x=0
    lat_base_smooth = (
        df_merged["Latency(ns)_Base"]
        .rolling(window, center=True, min_periods=1)
        .median()
        / 1e6
    )
    lat_mine_smooth = (
        df_merged["Latency(ns)_Mine"]
        .rolling(window, center=True, min_periods=1)
        .median()
        / 1e6
    )

    # Create the rich legend labels
    label_base = f"Baseline (EF={best_ef})  |  Avg Rec: {avg_rec_base:.4f}  |  Med Lat: {med_lat_base:.2f}ms"
    label_mine = f"Dynamic EF (Ours) |  Avg Rec: {avg_rec_mine:.4f}  |  Med Lat: {med_lat_mine:.2f}ms"

    ax1.plot(
        percentiles,
        lat_base_smooth,
        label=label_base,
        color="gray",
        linewidth=2,
        linestyle="--",
    )
    ax1.plot(percentiles, lat_mine_smooth, label=label_mine, color="blue", linewidth=3)
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title("Latency Profile Across Query Difficulty", fontsize=15, pad=15)

    # Place legend top right, with a solid white background
    ax1.legend(loc="upper right", fontsize=11, framealpha=1.0, edgecolor="black")
    ax1.grid(True, alpha=0.3)

    rec_base_smooth = (
        df_merged["Recall_Base"].rolling(window, center=True, min_periods=1).mean()
    )
    rec_mine_smooth = (
        df_merged["Recall_Mine"].rolling(window, center=True, min_periods=1).mean()
    )

    ax2.plot(
        percentiles,
        rec_base_smooth,
        label=label_base,
        color="gray",
        linewidth=2,
        linestyle="--",
    )
    ax2.plot(percentiles, rec_mine_smooth, label=label_mine, color="green", linewidth=3)
    ax2.set_ylabel("Recall", fontsize=12)
    ax2.set_xlabel(
        "Query Percentile (0 = Hardest Queries, 100 = Easiest Queries)", fontsize=13
    )
    ax2.set_title("Recall Profile Across Query Difficulty", fontsize=15, pad=15)

    # Place legend bottom right
    ax2.legend(loc="lower right", fontsize=11, framealpha=1.0, edgecolor="black")
    ax2.grid(True, alpha=0.3)

    # Adaptive Thresholds
    diff = (lat_mine_smooth - lat_base_smooth).dropna()
    cross_below = diff[diff <= 0]
    t1 = percentiles[cross_below.index[0]] if not cross_below.empty else 20

    cross_above = diff[(diff >= 0) & (percentiles[diff.index] < 80)]
    t2 = percentiles[cross_above.index[-1]] if not cross_above.empty else (t1 + 5)
    if t2 <= t1:
        t2 = t1 + 5

    c1 = "#ffcccc"
    c2 = "#ffffcc"
    c3 = "#ccffcc"
    for ax in [ax1, ax2]:
        ax.axvspan(0, t1, color=c1, alpha=0.4)
        ax.axvspan(t1, t2, color=c2, alpha=0.4)
        ax.axvspan(t2, 100, color=c3, alpha=0.4)

    bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)

    # Staggered Y-positions to prevent overlapping
    y_max1 = ax1.get_ylim()[1]
    y_min1 = ax1.get_ylim()[0]
    y_pos1_high = y_min1 + (y_max1 - y_min1) * 0.92
    y_pos1_low = y_min1 + (y_max1 - y_min1) * 0.78

    # X-positions (force Region 3 text to the left so it doesn't hit the legends)
    x1 = t1 / 2
    x2 = (t1 + t2) / 2
    x3 = (t2 + 100) / 2
    if x3 > 60:
        x3 = 60

    # Stagger Region 2 if it's too close to Region 1
    y_pos1_mid = y_pos1_low if (x2 - x1 < 15) else y_pos1_high

    ax1.text(
        x1,
        y_pos1_high,
        "Lower Recall Queries\n(Blue > Baseline)",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=bbox_props,
    )

    ax1.text(
        x2,
        y_pos1_mid,
        "Medium Recall Queries\n(Blue ≈ Baseline)",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=bbox_props,
    )

    ax1.text(
        x3,
        y_pos1_high,
        "High Recall Queries\n(Blue < Baseline)",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=bbox_props,
    )

    # Bottom Plot Text
    y_max2 = ax2.get_ylim()[1]
    y_min2 = ax2.get_ylim()[0]
    y_pos2_low = y_min2 + (y_max2 - y_min2) * 0.08
    y_pos2_high = y_min2 + (y_max2 - y_min2) * 0.22

    y_pos2_mid = y_pos2_high if (x2 - x1 < 15) else y_pos2_low

    ax2.text(
        x1,
        y_pos2_low,
        "(Green > Baseline)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=bbox_props,
    )

    ax2.text(
        x2,
        y_pos2_mid,
        "(Green ≈ Baseline)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=bbox_props,
    )

    ax2.text(
        x3,
        y_pos2_low,
        "(Green ≈ Baseline)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=bbox_props,
    )

    plt.tight_layout()
    out_path = f"/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story/story_continuous_final_{algo}_{dataset_name}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved Final Plot to: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--algo", choices=["shiro", "ada"], default="shiro", help="Algorithm to process")
    args = parser.parse_args()
    create_annotated_plot_with_stats(args.dataset, args.algo)
