import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def create_delta_plot(dataset_name, mine_csv, base_csv):
    print(f"Loading datasets for {dataset_name}...")
    df_mine = pd.read_csv(mine_csv)
    df_base = pd.read_csv(base_csv)

    summary_path = (
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv/summary_metrics.csv"
    )
    if os.path.exists(summary_path):
        sum_df = pd.read_csv(summary_path)
        ds_sum = sum_df[sum_df["dataset"] == dataset_name]

        # Check if the dataset is in the summary
        if not ds_sum.empty:
            ada_rows = ds_sum[ds_sum["method"] == "adaptive"]
            if not ada_rows.empty:
                adapt_rec = ada_rows["avg_recall"].values[0]
                adapt_lat = ada_rows["avg_lat(ns)"].values[0]
            else:
                adapt_rec = df_mine["Recall"].mean()
                adapt_lat = df_mine["Latency(ns)"].mean()

            base_df = ds_sum[ds_sum["method"] == "baseline"].copy()

            if not base_df.empty:
                base_df["ef"] = pd.to_numeric(base_df["ef"])
                base_df["recall_diff"] = (base_df["avg_recall"] - adapt_rec).abs()
                closest_ef = int(base_df.loc[base_df["recall_diff"].idxmin()]["ef"])
                closest_ef_lat = base_df.loc[base_df["recall_diff"].idxmin()][
                    "avg_lat(ns)"
                ]

                candidates = base_df[base_df["avg_recall"] < adapt_rec]
                if not candidates.empty:
                    best_ef = int(candidates.loc[candidates["ef"].idxmax()]["ef"])
                    best_ef_lat = base_df[base_df["ef"] == best_ef][
                        "avg_lat(ns)"
                    ].values[0]
                    if adapt_lat > best_ef_lat:
                        best_ef = closest_ef
                else:
                    best_ef = closest_ef
            else:
                # Fallback if no baseline in summary
                unique_efs = df_base["EF"].unique()
                ef_recalls = {
                    ef: df_base[df_base["EF"] == ef]["Recall"].mean()
                    for ef in unique_efs
                }
                candidates = {ef: r for ef, r in ef_recalls.items() if r < adapt_rec}
                if candidates:
                    best_ef = int(max(candidates, key=lambda ef: candidates[ef]))
                else:
                    best_ef = int(
                        min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - adapt_rec))
                    )
        else:
            adapt_rec = df_mine["Recall"].mean()
            unique_efs = df_base["EF"].unique()
            ef_recalls = {
                ef: df_base[df_base["EF"] == ef]["Recall"].mean() for ef in unique_efs
            }
            candidates = {ef: r for ef, r in ef_recalls.items() if r < adapt_rec}
            if candidates:
                best_ef = int(max(candidates, key=lambda ef: candidates[ef]))
            else:
                best_ef = int(
                    min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - adapt_rec))
                )
    else:
        adapt_rec = df_mine["Recall"].mean()

        # Find the highest EF where baseline recall < shiro recall
        unique_efs = df_base["EF"].unique()
        ef_recalls = {
            ef: df_base[df_base["EF"] == ef]["Recall"].mean() for ef in unique_efs
        }
        candidates = {ef: r for ef, r in ef_recalls.items() if r < adapt_rec}
        if candidates:
            best_ef = int(max(candidates, key=lambda ef: candidates[ef]))
        else:
            best_ef = int(
                min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - adapt_rec))
            )

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

    # Revert to minimal smoothing to bring back the "cool spikes"
    window = max(1, total_queries // 250)
    latency_diff = raw_lat_diff.rolling(window, center=True, min_periods=1).mean()
    recall_diff = raw_rec_diff.rolling(window, center=True, min_periods=1).mean()

    # --- SETUP PLOT ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

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
        f"Cost Impact: Latency Delta [{dataset_name}] (shiro-ef vs Baseline EF={best_ef})",
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
        0.95,
        roi_text_lat,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        zorder=10,
        bbox=dict(
            facecolor="#f8f9fa", alpha=0.95, edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )

    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.05),
        fontsize=12,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
    )

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
        f"Quality Impact: Recall Delta [{dataset_name}] (shiro-ef vs Baseline EF={best_ef})",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax2.set_ylabel("Recall Change", fontsize=14, fontweight="bold")
    ax2.set_xlabel(
        "Query Recall Percentile (0% = Lowest, 100% = Highest)",
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
        0.95,
        roi_text_rec,
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        zorder=10,
        bbox=dict(
            facecolor="#f8f9fa", alpha=0.95, edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )

    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}%"))
    ax2.set_xlim(0, 100)

    # Ensure symmetrical y-limits for recall delta to emphasize stability
    max_rec_diff = max(abs(recall_diff.max()), abs(recall_diff.min())) * 1.5
    if pd.isna(max_rec_diff) or max_rec_diff == 0:
        max_rec_diff = 0.05
    ax2.set_ylim(-max_rec_diff, max_rec_diff)
    ax2.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.05),
        fontsize=12,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
    )

    plt.tight_layout(pad=3.0)

    out_path = f"/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story/story_deltas_{dataset_name}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Delta Impact Plot to: {out_path}")

    plt.close()  # Close figure to avoid memory leaks when looping


if __name__ == "__main__":
    csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"
    for file in os.listdir(csv_dir):
        if file.startswith("per_query_results_") and file.endswith(".csv"):
            dataset_name = file.replace("per_query_results_", "").replace(".csv", "")
            mine_csv = os.path.join(csv_dir, file)
            base_csv = os.path.join(csv_dir, f"per_query_baseline_{dataset_name}.csv")

            if os.path.exists(base_csv):
                create_delta_plot(dataset_name, mine_csv, base_csv)
            else:
                print(
                    f"Warning: Baseline CSV not found for {dataset_name} ({base_csv})"
                )
