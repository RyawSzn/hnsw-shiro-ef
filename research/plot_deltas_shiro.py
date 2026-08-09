import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def generate_delta_plot():
    csv_shiro_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro"
    summary_path = os.path.join(csv_shiro_dir, "summary_metrics.csv")

    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Please run generate_summary_csv.py first.")
        return

    sum_df = pd.read_csv(summary_path)
    all_datasets = sorted(sum_df["dataset"].unique())

    if len(all_datasets) == 0:
        print("No datasets found in summary_metrics.csv")
        return

    print("Available datasets:")
    for i, ds in enumerate(all_datasets):
        print(f"  [{i}] {ds}")

    user_input = input(
        "\nEnter dataset numbers to plot (comma-separated), or press Enter for all: "
    ).strip()

    if not user_input or user_input.lower() == "all":
        datasets = all_datasets
    else:
        selected_indices = []
        for p in user_input.split(","):
            p = p.strip()
            if p.isdigit():
                idx = int(p)
                if 0 <= idx < len(all_datasets):
                    selected_indices.append(idx)
                else:
                    print(f"Warning: Index {idx} out of bounds, skipping.")
            else:
                print(f"Warning: Invalid input '{p}', skipping.")

        datasets = [all_datasets[i] for i in dict.fromkeys(selected_indices)]

    num_ds = len(datasets)
    if num_ds == 0:
        print("No datasets selected. Exiting.")
        return

    ncols = min(num_ds, 6)
    nrows = math.ceil(num_ds / ncols) * 2  # 2 rows (latency + recall) per dataset row
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * math.ceil(num_ds / ncols)))

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Hide unused axes
    total_axes = nrows * ncols
    # We use pairs: ax_lat = axes_flat[2*plot_col], ax_rec = axes_flat[2*plot_col+1]
    # Restructure: treat axes as (nrows, ncols) where rows alternate lat/rec per dataset row
    axes_2d = np.array(axes_flat).reshape(nrows, ncols)

    for plot_idx, ds_name in enumerate(datasets):
        col = plot_idx % ncols
        row_pair = (plot_idx // ncols) * 2  # top row of the lat/rec pair

        ax1 = axes_2d[row_pair, col]       # latency delta
        ax2 = axes_2d[row_pair + 1, col]   # recall delta

        # Per-query CSVs
        base_csv = os.path.join(csv_shiro_dir, f"per_query_baseline_{ds_name}.csv")
        shiro_csv = os.path.join(csv_shiro_dir, f"per_query_results_{ds_name}.csv")

        if not os.path.exists(base_csv):
            ax1.set_visible(False)
            ax2.set_visible(False)
            print(f"Warning: baseline CSV not found for {ds_name}, skipping.")
            continue
        if not os.path.exists(shiro_csv):
            ax1.set_visible(False)
            ax2.set_visible(False)
            print(f"Warning: shiro CSV not found for {ds_name}, skipping.")
            continue

        df_base = pd.read_csv(base_csv)
        df_shiro = pd.read_csv(shiro_csv)

        adapt_rec = df_shiro["Recall"].mean()
        adapt_lat = df_shiro["Latency(ns)"].mean()

        # Find the best comparable baseline EF from summary
        best_ef = None
        ds_sum = sum_df[sum_df["dataset"] == ds_name]
        if not ds_sum.empty:
            base_sum = ds_sum[ds_sum["method"] == "baseline"].copy()
            if not base_sum.empty:
                base_sum["ef"] = pd.to_numeric(base_sum["ef"])
                base_sum["recall_diff"] = (base_sum["avg_recall"] - adapt_rec).abs()
                closest_ef = int(base_sum.loc[base_sum["recall_diff"].idxmin()]["ef"])

                candidates = base_sum[base_sum["avg_recall"] < adapt_rec]
                if not candidates.empty:
                    best_ef_cand = int(candidates.loc[candidates["ef"].idxmax()]["ef"])
                    best_ef_lat = base_sum[base_sum["ef"] == best_ef_cand]["avg_lat(ns)"].values[0]
                    best_ef = closest_ef if adapt_lat > best_ef_lat else best_ef_cand
                else:
                    best_ef = closest_ef

        if best_ef is None:
            unique_efs = df_base["EF"].unique()
            ef_recalls = {ef: df_base[df_base["EF"] == ef]["Recall"].mean() for ef in unique_efs}
            candidates = {ef: r for ef, r in ef_recalls.items() if r < adapt_rec}
            best_ef = (
                int(max(candidates, key=lambda ef: candidates[ef]))
                if candidates
                else int(min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - adapt_rec)))
            )

        df_base_best = df_base[df_base["EF"] == best_ef].copy()
        df_merged = pd.merge(df_base_best, df_shiro, on="QueryID", suffixes=("_Base", "_Mine"))
        df_merged.sort_values(
            ["Recall_Base", "Latency(ns)_Base"], ascending=[True, False], inplace=True
        )
        df_merged.reset_index(drop=True, inplace=True)

        total_queries = len(df_merged)
        x_percentile = np.linspace(0, 100, total_queries)

        raw_lat_diff = df_merged["Latency(ns)_Mine"] / 1e6 - df_merged["Latency(ns)_Base"] / 1e6
        lat_red_area = raw_lat_diff[raw_lat_diff > 0].sum()
        lat_green_area = -raw_lat_diff[raw_lat_diff < 0].sum()
        lat_net_profit = lat_green_area - lat_red_area

        raw_rec_diff = df_merged["Recall_Mine"] - df_merged["Recall_Base"]
        rec_green_area = raw_rec_diff[raw_rec_diff > 0].sum()
        rec_red_area = -raw_rec_diff[raw_rec_diff < 0].sum()
        rec_net_profit = rec_green_area - rec_red_area

        window = max(1, total_queries // 500)
        latency_diff = raw_lat_diff.rolling(window, center=True, min_periods=1).mean()
        recall_diff = raw_rec_diff.rolling(window, center=True, min_periods=1).mean()

        # --- LATENCY DELTA ---
        ax1.plot(x_percentile, latency_diff, color="black", linewidth=1, alpha=0.5)
        ax1.fill_between(
            x_percentile, latency_diff, 0,
            where=(latency_diff > 0), interpolate=True,
            color="#e74c3c", alpha=0.7, label="Compute Investment (+ Latency)",
        )
        ax1.fill_between(
            x_percentile, latency_diff, 0,
            where=(latency_diff <= 0), interpolate=True,
            color="#2ecc71", alpha=0.7, label="Compute Savings (- Latency)",
        )
        ax1.axhline(0, color="black", linewidth=1.5, linestyle="--")
        ax1.set_title(
            f"{ds_name}\nLatency Delta vs Baseline EF={best_ef}",
            fontsize=12, fontweight="bold", pad=10,
        )
        ax1.set_ylabel("Latency Change (ms)", fontsize=12)
        ax1.text(
            0.98, 0.95,
            (
                f"Compute ROI:\n"
                f"Savings: {lat_green_area:,.1f} ms\n"
                f"Investment: -{lat_red_area:,.1f} ms\n"
                f"────────────\n"
                f"Net: {lat_net_profit:,.1f} ms"
            ),
            transform=ax1.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", zorder=10,
            bbox=dict(facecolor="#f8f9fa", alpha=0.95, edgecolor="black", boxstyle="round,pad=0.4"),
        )

        # --- RECALL DELTA ---
        ax2.plot(x_percentile, recall_diff, color="black", linewidth=1, alpha=0.5)
        ax2.fill_between(
            x_percentile, recall_diff, 0,
            where=(recall_diff > 0), interpolate=True,
            color="#2ecc71", alpha=0.7, label="Accuracy Rescued (+ Recall)",
        )
        ax2.fill_between(
            x_percentile, recall_diff, 0,
            where=(recall_diff <= 0), interpolate=True,
            color="#e74c3c", alpha=0.7, label="Accuracy Dropped (- Recall)",
        )
        ax2.axhline(0, color="black", linewidth=1.5, linestyle="--")
        ax2.set_title(
            f"Recall Delta vs Baseline EF={best_ef}",
            fontsize=12, fontweight="bold", pad=10,
        )
        ax2.set_ylabel("Recall Change", fontsize=12)
        ax2.set_xlabel(
            "Query Recall Percentile (0% = Lowest, 100% = Highest)",
            fontsize=11, labelpad=8,
        )
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}%"))
        ax2.set_xlim(0, 100)
        ax2.text(
            0.98, 0.95,
            (
                f"Accuracy ROI:\n"
                f"Gained: +{rec_green_area:,.3f}\n"
                f"Lost: -{rec_red_area:,.3f}\n"
                f"────────────\n"
                f"Net: {rec_net_profit:,.3f}"
            ),
            transform=ax2.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", zorder=10,
            bbox=dict(facecolor="#f8f9fa", alpha=0.95, edgecolor="black", boxstyle="round,pad=0.4"),
        )

    # Hide leftover axes
    for plot_idx in range(num_ds, ncols * (nrows // 2)):
        col = plot_idx % ncols
        row_pair = (plot_idx // ncols) * 2
        axes_2d[row_pair, col].set_visible(False)
        axes_2d[row_pair + 1, col].set_visible(False)

    fig.suptitle("Shiro-EF vs Baseline: Delta Impact Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout(pad=1.5)

    out_path = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/delta_shiro.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    generate_delta_plot()
