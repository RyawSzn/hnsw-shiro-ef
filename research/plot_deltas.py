import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def create_delta_plot():
    print("Loading datasets...")
    df_mine = pd.read_csv(
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv/per_query_results_deep-image-96-angular.csv"
    )
    df_base = pd.read_csv(
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv/per_query_baseline_deep-image-96-angular.csv"
    )

    # Match average recall to find best baseline EF
    avg_recall_mine = df_mine["Recall"].mean()
    if 450 in df_base["EF"].unique():
        best_ef = 450
    else:
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
    df_merged.sort_values(
        ["Recall_Base", "Latency(ns)_Base"], ascending=[True, False], inplace=True
    )
    df_merged.reset_index(drop=True, inplace=True)

    total_queries = len(df_merged)
    x_percentile = np.linspace(0, 100, total_queries)

    # Smooth the deltas for a clean business visualization
    window = max(1, total_queries // 50)

    # Calculate exact mathematical areas (using raw data, not smoothed, for accuracy)
    raw_lat_diff = (
        df_merged["Latency(ns)_Mine"] / 1e6 - df_merged["Latency(ns)_Base"] / 1e6
    )
    lat_red_area = raw_lat_diff[raw_lat_diff > 0].sum()
    lat_green_area = -raw_lat_diff[raw_lat_diff < 0].sum()  # Make positive magnitude
    lat_net_profit = lat_green_area - lat_red_area

    raw_rec_diff = df_merged["Recall_Mine"] - df_merged["Recall_Base"]
    rec_green_area = raw_rec_diff[raw_rec_diff > 0].sum()
    rec_red_area = -raw_rec_diff[raw_rec_diff < 0].sum()  # Make positive magnitude
    rec_net_profit = rec_green_area - rec_red_area

    latency_diff = raw_lat_diff.rolling(window, center=True, min_periods=1).mean()
    recall_diff = raw_rec_diff.rolling(window, center=True, min_periods=1).mean()

    # --- SETUP PLOT ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # Zone Definitions
    zone1_end = 27
    zone2_end = 37

    # --- TOP PLOT: LATENCY DELTA ---
    ax1.plot(x_percentile, latency_diff, color="black", linewidth=1, alpha=0.5)

    # Fill red for investment (Shiro > Base)
    ax1.fill_between(
        x_percentile,
        latency_diff,
        0,
        where=(latency_diff > 0),
        interpolate=True,
        color="#e74c3c",
        alpha=0.7,
        label="Compute Investment (+ Latency)",
    )

    # Fill green for savings (Shiro < Base)
    ax1.fill_between(
        x_percentile,
        latency_diff,
        0,
        where=(latency_diff <= 0),
        interpolate=True,
        color="#2ecc71",
        alpha=0.7,
        label="Compute Savings (- Latency)",
    )

    ax1.axhline(0, color="black", linewidth=1.5, linestyle="--")
    ax1.set_title(
        f"Cost Impact: Latency Delta (shiro-ef vs Baseline EF={best_ef})",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.set_ylabel("Latency Change (ms)", fontsize=14, fontweight="bold")

    # Add ROI Text Box to Latency Plot
    roi_text_lat = (
        f"Compute ROI (per {total_queries:,} queries):\n"
        f"Total Savings (Green): {lat_green_area:,.1f} ms\n"
        f"Total Investment (Red): -{lat_red_area:,.1f} ms\n"
        f"──────────────────────\n"
        f"Net Profit (Green - Red): +{lat_net_profit:,.1f} ms"
    )

    ax1.text(
        0.98,
        0.05,
        roi_text_lat,
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            facecolor="#f8f9fa", alpha=0.9, edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )

    ax1.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95), fontsize=12)

    # --- BOTTOM PLOT: RECALL DELTA ---
    ax2.plot(x_percentile, recall_diff, color="black", linewidth=1, alpha=0.5)

    # Fill green for accuracy gained (Shiro > Base)
    ax2.fill_between(
        x_percentile,
        recall_diff,
        0,
        where=(recall_diff > 0),
        interpolate=True,
        color="#2ecc71",
        alpha=0.7,
        label="Accuracy Rescued (+ Recall)",
    )

    # Fill red for accuracy lost (Shiro < Base)
    ax2.fill_between(
        x_percentile,
        recall_diff,
        0,
        where=(recall_diff <= 0),
        interpolate=True,
        color="#e74c3c",
        alpha=0.7,
        label="Accuracy Dropped (- Recall)",
    )

    ax2.axhline(0, color="black", linewidth=1.5, linestyle="--")
    ax2.set_title(
        f"Quality Impact: Recall Delta (shiro-ef vs Baseline EF={best_ef})",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax2.set_ylabel("Recall Change", fontsize=14, fontweight="bold")
    ax2.set_xlabel(
        "Query Difficulty Percentile (0% = Hardest, 100% = Easiest)",
        fontsize=14,
        fontweight="bold",
        labelpad=10,
    )

    # Add ROI Text Box to Recall Plot
    roi_text_rec = (
        f"Accuracy ROI (per {total_queries:,} queries):\n"
        f"Accuracy Gained (Green): +{rec_green_area:,.1f}\n"
        f"Accuracy Lost (Red): -{rec_red_area:,.1f}\n"
        f"──────────────────────\n"
        f"Net Profit (Green - Red): +{rec_net_profit:,.1f}"
    )

    ax2.text(
        0.98,
        0.05,
        roi_text_rec,
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            facecolor="#f8f9fa", alpha=0.9, edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )

    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}%"))
    ax2.set_xlim(0, 100)

    # Ensure symmetrical y-limits for recall delta to emphasize stability
    max_rec_diff = max(abs(recall_diff.max()), abs(recall_diff.min())) * 1.5
    if pd.isna(max_rec_diff) or max_rec_diff == 0:
        max_rec_diff = 0.05
    ax2.set_ylim(-max_rec_diff, max_rec_diff)
    ax2.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95), fontsize=12)

    plt.tight_layout(pad=3.0)

    out_path = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story/story_deltas.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Delta Impact Plot to: {out_path}")


if __name__ == "__main__":
    create_delta_plot()
