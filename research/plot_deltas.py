import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def create_delta_plot(dataset_name, base_csv, shiro_csv, ada_csv):
    print(f"Loading datasets for {dataset_name}...")
    df_base = pd.read_csv(base_csv)

    df_shiro = pd.read_csv(shiro_csv) if os.path.exists(shiro_csv) else None
    df_ada = pd.read_csv(ada_csv) if os.path.exists(ada_csv) else None
    
    if df_shiro is None and df_ada is None:
        print("Both shiro and ada CSVs missing.")
        return

    # Use shiro as reference for finding best EF if available, else ada
    ref_df = df_shiro if df_shiro is not None else df_ada
    adapt_rec = ref_df["Recall"].mean()
    adapt_lat = ref_df["Latency(ns)"].mean()

    summary_path = (
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro/summary_metrics.csv"
    )
    if os.path.exists(summary_path):
        sum_df = pd.read_csv(summary_path)
        ds_sum = sum_df[sum_df["dataset"] == dataset_name]

        if not ds_sum.empty:
            base_df = ds_sum[ds_sum["method"] == "baseline"].copy()
            if not base_df.empty:
                base_df["ef"] = pd.to_numeric(base_df["ef"])
                base_df["recall_diff"] = (base_df["avg_recall"] - adapt_rec).abs()
                closest_ef = int(base_df.loc[base_df["recall_diff"].idxmin()]["ef"])
                
                candidates = base_df[base_df["avg_recall"] < adapt_rec]
                if not candidates.empty:
                    best_ef = int(candidates.loc[candidates["ef"].idxmax()]["ef"])
                    best_ef_lat = base_df[base_df["ef"] == best_ef]["avg_lat(ns)"].values[0]
                    if adapt_lat > best_ef_lat:
                        best_ef = closest_ef
                else:
                    best_ef = closest_ef
            else:
                unique_efs = df_base["EF"].unique()
                ef_recalls = {ef: df_base[df_base["EF"] == ef]["Recall"].mean() for ef in unique_efs}
                candidates = {ef: r for ef, r in ef_recalls.items() if r < adapt_rec}
                best_ef = int(max(candidates, key=lambda ef: candidates[ef])) if candidates else int(min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - adapt_rec)))
        else:
            unique_efs = df_base["EF"].unique()
            ef_recalls = {ef: df_base[df_base["EF"] == ef]["Recall"].mean() for ef in unique_efs}
            candidates = {ef: r for ef, r in ef_recalls.items() if r < adapt_rec}
            best_ef = int(max(candidates, key=lambda ef: candidates[ef])) if candidates else int(min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - adapt_rec)))
    else:
        unique_efs = df_base["EF"].unique()
        ef_recalls = {ef: df_base[df_base["EF"] == ef]["Recall"].mean() for ef in unique_efs}
        candidates = {ef: r for ef, r in ef_recalls.items() if r < adapt_rec}
        best_ef = int(max(candidates, key=lambda ef: candidates[ef])) if candidates else int(min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - adapt_rec)))

    df_base_best = df_base[df_base["EF"] == best_ef].copy()
    
    # Setup subplots based on what's available
    num_cols = sum(x is not None for x in [df_shiro, df_ada])
    if num_cols == 0:
        return
        
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(2, num_cols, figsize=(8 * num_cols, 12), sharex=True, sharey='row', squeeze=False)
    
    col_idx = 0
    for algo_name, df_mine in [("shiro", df_shiro), ("ada", df_ada)]:
        if df_mine is None:
            continue
            
        df_merged = pd.merge(df_base_best, df_mine, on="QueryID", suffixes=("_Base", "_Mine"))
        df_merged.sort_values(["Recall_Base", "Latency(ns)_Base"], ascending=[True, False], inplace=True)
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

        window = max(1, total_queries // 250)
        latency_diff = raw_lat_diff.rolling(window, center=True, min_periods=1).mean()
        recall_diff = raw_rec_diff.rolling(window, center=True, min_periods=1).mean()

        ax1 = axs[0, col_idx]
        ax2 = axs[1, col_idx]
        
        # --- TOP PLOT: LATENCY DELTA ---
        ax1.plot(x_percentile, latency_diff, color="black", linewidth=1, alpha=0.5)
        ax1.fill_between(x_percentile, latency_diff, 0, where=(latency_diff > 0), interpolate=True, color="#e74c3c", alpha=0.7, label="Compute Investment (+ Latency)")
        ax1.fill_between(x_percentile, latency_diff, 0, where=(latency_diff <= 0), interpolate=True, color="#2ecc71", alpha=0.7, label="Compute Savings (- Latency)")
        ax1.axhline(0, color="black", linewidth=1.5, linestyle="--")
        
        ax1.set_title(f"[{algo_name}-ef] Latency Delta vs Baseline EF={best_ef}", fontsize=14, fontweight="bold", pad=15)
        if col_idx == 0:
            ax1.set_ylabel("Latency Change (ms)", fontsize=14, fontweight="bold")

        roi_text_lat = (
            f"Compute ROI ({algo_name}):\n"
            f"Savings (Green): {lat_green_area:,.1f} ms\n"
            f"Investment (Red): -{lat_red_area:,.1f} ms\n"
            f"────────────────\n"
            f"Net Profit: {lat_net_profit:,.1f} ms"
        )
        ax1.text(0.98, 0.95, roi_text_lat, transform=ax1.transAxes, ha="right", va="top", fontsize=11, fontweight="bold", zorder=10, bbox=dict(facecolor="#f8f9fa", alpha=0.95, edgecolor="black", boxstyle="round,pad=0.5"))
        if col_idx == 0:
            ax1.legend(loc="lower left", fontsize=11, frameon=True, facecolor="white", framealpha=0.95)

        # --- BOTTOM PLOT: RECALL DELTA ---
        ax2.plot(x_percentile, recall_diff, color="black", linewidth=1, alpha=0.5)
        ax2.fill_between(x_percentile, recall_diff, 0, where=(recall_diff > 0), interpolate=True, color="#2ecc71", alpha=0.7, label="Accuracy Rescued (+ Recall)")
        ax2.fill_between(x_percentile, recall_diff, 0, where=(recall_diff <= 0), interpolate=True, color="#e74c3c", alpha=0.7, label="Accuracy Dropped (- Recall)")
        ax2.axhline(0, color="black", linewidth=1.5, linestyle="--")
        
        ax2.set_title(f"[{algo_name}-ef] Recall Delta vs Baseline EF={best_ef}", fontsize=14, fontweight="bold", pad=15)
        if col_idx == 0:
            ax2.set_ylabel("Recall Change", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Query Recall Percentile (0% = Lowest, 100% = Highest)", fontsize=13, fontweight="bold", labelpad=10)

        roi_text_rec = (
            f"Accuracy ROI ({algo_name}):\n"
            f"Gained (Green): +{rec_green_area:,.1f}\n"
            f"Lost (Red): -{rec_red_area:,.1f}\n"
            f"────────────────\n"
            f"Net Profit: {rec_net_profit:,.1f}"
        )
        ax2.text(0.98, 0.95, roi_text_rec, transform=ax2.transAxes, ha="right", va="top", fontsize=11, fontweight="bold", zorder=10, bbox=dict(facecolor="#f8f9fa", alpha=0.95, edgecolor="black", boxstyle="round,pad=0.5"))
        
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}%"))
        ax2.set_xlim(0, 100)
        
        if col_idx == 0:
            ax2.legend(loc="lower left", fontsize=11, frameon=True, facecolor="white", framealpha=0.95)
            
        col_idx += 1
        
    # Ensure symmetrical y-limits for recall delta across all columns by finding max across both
    all_recall_lines = [line.get_ydata() for ax in axs[1,:] for line in ax.get_lines() if len(line.get_ydata()) > 2]
    if all_recall_lines:
        max_rec_diff = max(np.nanmax(np.abs(y)) for y in all_recall_lines) * 1.5
        if pd.isna(max_rec_diff) or max_rec_diff == 0:
            max_rec_diff = 0.05
        axs[1,0].set_ylim(-max_rec_diff, max_rec_diff)

    fig.suptitle(f"Delta Impact Analysis: {dataset_name}", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])

    out_path = f"/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story/story_deltas_combined_{dataset_name}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Combined Delta Impact Plot to: {out_path}")

    plt.close()


if __name__ == "__main__":
    csv_shiro_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro"
    csv_ada_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_ada"
    
    for file in os.listdir(csv_shiro_dir):
        if file.startswith("per_query_baseline_") and file.endswith(".csv"):
            dataset_name = file.replace("per_query_baseline_", "").replace(".csv", "")
            base_csv = os.path.join(csv_shiro_dir, file)
            shiro_csv = os.path.join(csv_shiro_dir, f"per_query_results_{dataset_name}.csv")
            ada_csv = os.path.join(csv_ada_dir, f"per_query_results_{dataset_name}.csv")

            if os.path.exists(base_csv):
                create_delta_plot(dataset_name, base_csv, shiro_csv, ada_csv)
            else:
                print(f"Warning: Baseline CSV not found for {dataset_name} ({base_csv})")
