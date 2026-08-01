import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


def create_consistency_plot(dataset_name, mine_csv, base_csv):
    print(f"Loading datasets for {dataset_name}...")
    df_mine = pd.read_csv(mine_csv)
    df_base = pd.read_csv(base_csv)

    summary_path = (
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv/summary_metrics.csv"
    )
    if os.path.exists(summary_path):
        sum_df = pd.read_csv(summary_path)
        ds_sum = sum_df[sum_df["dataset"] == dataset_name]
        adapt_rec = ds_sum[ds_sum["method"] == "adaptive"]["avg_recall"].values[0]

        base_df = ds_sum[ds_sum["method"] == "baseline"].copy()
        base_df["ef"] = pd.to_numeric(base_df["ef"])
        valid_bases = base_df[base_df["avg_recall"] < adapt_rec]

        if not valid_bases.empty:
            best_ef = int(valid_bases["ef"].max())
        else:
            best_ef = int(base_df["ef"].min())
    else:
        adapt_rec = df_mine["Recall"].mean()
        valid_efs = [
            ef
            for ef in df_base["EF"].unique()
            if df_base[df_base["EF"] == ef]["Recall"].mean() < adapt_rec
        ]
        best_ef = int(max(valid_efs)) if valid_efs else int(df_base["EF"].min())

    df_base_best = df_base[df_base["EF"] == best_ef].copy()
    df_merged = pd.merge(
        df_base_best, df_mine, on="QueryID", suffixes=("_Base", "_Mine")
    )

    # Calculate key stats
    std_base = df_merged["Recall_Base"].std()
    std_mine = df_merged["Recall_Mine"].std()
    p01_base = df_merged["Recall_Base"].quantile(0.01)
    p01_mine = df_merged["Recall_Mine"].quantile(0.01)

    # --- SETUP PLOT ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    color_base = "#7f8c8d"  # Gray
    color_shiro = "#27ae60"  # Green

    # --- PANEL 1: VIOLIN PLOT (Distribution of Quality) ---
    data = [df_merged["Recall_Base"], df_merged["Recall_Mine"]]

    parts = ax1.violinplot(data, showmeans=True, showextrema=True, bw_method=0.15)

    # Color customization for violin
    parts["bodies"][0].set_facecolor(color_base)
    parts["bodies"][1].set_facecolor(color_shiro)
    for p in ["bodies"]:
        for body in parts[p]:
            body.set_alpha(0.7)
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1.5)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(
        [f"Baseline (Fixed EF)", f"shiro-ef (Dynamic)"], fontsize=12, fontweight="bold"
    )
    ax1.set_ylabel("Recall Accuracy", fontsize=14, fontweight="bold")
    ax1.set_title(
        f"Variance / Distribution of Quality [{dataset_name}]",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    # Annotate Variance and Worst-Case Floor
    ax1.annotate(
        f"Long tail of failure\nWorst 1%: {p01_base:.3f}\nStdDev: {std_base:.3f}",
        xy=(1, p01_base),
        xytext=(50, -30),
        textcoords="offset points",
        ha="left",
        color="black",
        fontweight="bold",
        fontsize=11,
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
    )

    ax1.annotate(
        f"Tight, consistent quality\nWorst 1%: {p01_mine:.3f}\nStdDev: {std_mine:.3f}",
        xy=(2, p01_mine),
        xytext=(50, -30),
        textcoords="offset points",
        ha="left",
        color="black",
        fontweight="bold",
        fontsize=11,
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
    )

    ax1.set_ylim(0.4, 1.05)

    # --- PANEL 2: SLA GUARANTEE (Survival Curve) ---
    # Sort data for survival curve
    sorted_base = np.sort(df_merged["Recall_Base"])
    sorted_mine = np.sort(df_merged["Recall_Mine"])

    # Calculate % of queries >= X
    yvals_base = 1.0 - np.arange(len(sorted_base)) / (len(sorted_base) - 1)
    yvals_mine = 1.0 - np.arange(len(sorted_mine)) / (len(sorted_mine) - 1)

    ax2.plot(
        sorted_base,
        yvals_base * 100,
        color=color_base,
        linewidth=4,
        linestyle="--",
        label="Baseline",
    )
    ax2.plot(
        sorted_mine, yvals_mine * 100, color=color_shiro, linewidth=4, label="shiro-ef"
    )

    ax2.set_xlim(0.80, 1.01)
    ax2.set_ylim(50, 105)

    # Add a target SLA line (e.g., 90% Recall)
    target_recall = 0.90
    pct_base_pass = (sorted_base >= target_recall).mean() * 100
    pct_mine_pass = (sorted_mine >= target_recall).mean() * 100

    ax2.axvline(target_recall, color="black", linestyle=":", linewidth=2, alpha=0.5)
    ax2.plot(target_recall, pct_base_pass, marker="o", color=color_base, markersize=10)
    ax2.plot(target_recall, pct_mine_pass, marker="o", color=color_shiro, markersize=10)

    ax2.annotate(
        f"Base: {pct_base_pass:.1f}%",
        xy=(target_recall, pct_base_pass),
        xytext=(-15, -15),
        textcoords="offset points",
        ha="right",
        color=color_base,
        fontweight="bold",
        fontsize=11,
    )

    ax2.annotate(
        f"Shiro: {pct_mine_pass:.1f}%",
        xy=(target_recall, pct_mine_pass),
        xytext=(15, 10),
        textcoords="offset points",
        ha="left",
        color=color_shiro,
        fontweight="bold",
        fontsize=11,
    )

    # Add the experimental target recall line (0.95)
    target_exp = 0.95
    pct_base_pass_exp = (sorted_base >= target_exp).mean() * 100
    pct_mine_pass_exp = (sorted_mine >= target_exp).mean() * 100

    ax2.axvline(target_exp, color="#e74c3c", linestyle=":", linewidth=2.5, alpha=0.8)
    ax2.plot(target_exp, pct_base_pass_exp, marker="o", color=color_base, markersize=10)
    ax2.plot(target_exp, pct_mine_pass_exp, marker="o", color=color_shiro, markersize=10)

    ax2.annotate(
        f"Base: {pct_base_pass_exp:.1f}%",
        xy=(target_exp, pct_base_pass_exp),
        xytext=(-15, -15),
        textcoords="offset points",
        ha="right",
        color=color_base,
        fontweight="bold",
        fontsize=11,
    )

    ax2.annotate(
        f"Shiro: {pct_mine_pass_exp:.1f}%",
        xy=(target_exp, pct_mine_pass_exp),
        xytext=(15, 10),
        textcoords="offset points",
        ha="left",
        color=color_shiro,
        fontweight="bold",
        fontsize=11,
    )

    ax2.set_xlabel("Target Recall Guarantee (SLA)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("% of Queries Achieving Target", fontsize=14, fontweight="bold")
    ax2.set_title(
        f"Reliability: Service Level Agreement (SLA) [{dataset_name}]",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.legend(loc="lower left", fontsize=12)

    plt.tight_layout(pad=3.0)

    out_path = f"/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story/story_consistency_visuals_{dataset_name}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Consistency Visuals to: {out_path}")

    plt.close()  # Close figure to avoid memory leaks


if __name__ == "__main__":
    csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"
    for file in os.listdir(csv_dir):
        if file.startswith("per_query_results_") and file.endswith(".csv"):
            dataset_name = file.replace("per_query_results_", "").replace(".csv", "")
            mine_csv = os.path.join(csv_dir, file)
            base_csv = os.path.join(csv_dir, f"per_query_baseline_{dataset_name}.csv")

            if os.path.exists(base_csv):
                create_consistency_plot(dataset_name, mine_csv, base_csv)
            else:
                print(
                    f"Warning: Baseline CSV not found for {dataset_name} ({base_csv})"
                )
