import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


def create_consistency_plot(dataset_name, base_csv, shiro_csv, ada_csv):
    print(f"Loading datasets for {dataset_name}...")
    df_base = pd.read_csv(base_csv)
    
    df_shiro = pd.read_csv(shiro_csv) if os.path.exists(shiro_csv) else None
    df_ada = pd.read_csv(ada_csv) if os.path.exists(ada_csv) else None

    # Determine target recall from shiro (or ada if shiro is missing)
    if df_shiro is not None:
        target_rec = df_shiro["Recall"].mean()
    elif df_ada is not None:
        target_rec = df_ada["Recall"].mean()
    else:
        print("Both shiro and ada CSVs are missing.")
        return

    summary_path = (
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro/summary_metrics.csv"
    )
    if os.path.exists(summary_path):
        sum_df = pd.read_csv(summary_path)
        ds_sum = sum_df[sum_df["dataset"] == dataset_name]
        
        base_df = ds_sum[ds_sum["method"] == "baseline"].copy()
        if not base_df.empty:
            base_df["ef"] = pd.to_numeric(base_df["ef"])
            base_df["recall_diff"] = (base_df["avg_recall"] - target_rec).abs()
            best_ef = int(base_df.loc[base_df["recall_diff"].idxmin()]["ef"])
        else:
            unique_efs = df_base["EF"].unique()
            ef_recalls = {ef: df_base[df_base["EF"] == ef]["Recall"].mean() for ef in unique_efs}
            best_ef = int(min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - target_rec)))
    else:
        unique_efs = df_base["EF"].unique()
        ef_recalls = {ef: df_base[df_base["EF"] == ef]["Recall"].mean() for ef in unique_efs}
        best_ef = int(min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - target_rec)))

    df_base_best = df_base[df_base["EF"] == best_ef].copy()
    
    # Merge dataframes
    df_merged = df_base_best.rename(columns={"Recall": "Recall_Base", "Latency(ns)": "Latency_Base"})
    
    if df_shiro is not None:
        df_merged = pd.merge(df_merged, df_shiro[["QueryID", "Recall", "Latency(ns)"]].rename(columns={"Recall": "Recall_Shiro", "Latency(ns)": "Latency_Shiro"}), on="QueryID")
    if df_ada is not None:
        df_merged = pd.merge(df_merged, df_ada[["QueryID", "Recall", "Latency(ns)"]].rename(columns={"Recall": "Recall_Ada", "Latency(ns)": "Latency_Ada"}), on="QueryID")

    # --- SETUP PLOT ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    color_base = "#7f8c8d"  # Gray
    color_shiro = "#27ae60"  # Green
    color_ada = "#2980b9"    # Blue

    # --- PANEL 1: VIOLIN PLOT (Distribution of Quality) ---
    data = [df_merged["Recall_Base"]]
    labels = [f"Baseline (EF={best_ef})"]
    colors = [color_base]
    
    if df_shiro is not None:
        data.append(df_merged["Recall_Shiro"])
        labels.append("shiro-ef")
        colors.append(color_shiro)
        
    if df_ada is not None:
        data.append(df_merged["Recall_Ada"])
        labels.append("ada-ef")
        colors.append(color_ada)

    parts = ax1.violinplot(data, showmeans=True, showextrema=True, bw_method=0.15)

    # Color customization for violin
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1.5)

    ax1.set_xticks(np.arange(1, len(labels) + 1))
    ax1.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Recall Accuracy", fontsize=14, fontweight="bold")
    ax1.set_title(
        f"Variance / Distribution of Quality [{dataset_name}]",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    # Annotate Variance and Worst-Case Floor
    for i, (series, label, c) in enumerate(zip(data, labels, colors)):
        p01 = series.quantile(0.01)
        std = series.std()
        
        text = f"Worst 1%: {p01:.3f}\nStdDev: {std:.3f}"
        if i == 0:
            text = "Long tail of failure\n" + text
            xytext = (40, -30)
        else:
            text = "Tight quality\n" + text
            xytext = (40, -30) if i == 1 else (40, 30)

        ax1.annotate(
            text,
            xy=(i + 1, p01),
            xytext=xytext,
            textcoords="offset points",
            ha="left",
            color="black",
            fontweight="bold",
            fontsize=11,
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        )

    ax1.set_ylim(0.4, 1.05)

    # --- PANEL 2: SLA GUARANTEE (Survival Curve) ---
    for series, label, c, style in zip(data, labels, colors, ["--", "-", "-"]):
        sorted_vals = np.sort(series)
        yvals = 1.0 - np.arange(len(sorted_vals)) / (len(sorted_vals) - 1)
        ax2.plot(
            sorted_vals,
            yvals * 100,
            color=c,
            linewidth=4 if style == "-" else 3,
            linestyle=style,
            label=label,
        )

    ax2.set_xlim(0.80, 1.01)
    ax2.set_ylim(50, 105)

    # Add a target SLA line (e.g., 0.90 and 0.95)
    for target, offset_y in [(0.90, -15), (0.95, -30)]:
        line_color = "black" if target == 0.90 else "#e74c3c"
        ax2.axvline(target, color=line_color, linestyle=":", linewidth=2, alpha=0.5)
        
        for i, (series, label, c) in enumerate(zip(data, labels, colors)):
            pct_pass = (series >= target).mean() * 100
            ax2.plot(target, pct_pass, marker="o", color=c, markersize=10)
            
            # Stagger text to avoid overlap
            if i == 0:
                xytext = (-15, -15)
                ha = "right"
            elif i == 1:
                xytext = (15, 10)
                ha = "left"
            else:
                xytext = (15, -20)
                ha = "left"
                
            ax2.annotate(
                f"{label.split(' ')[0]}: {pct_pass:.1f}%",
                xy=(target, pct_pass),
                xytext=xytext,
                textcoords="offset points",
                ha=ha,
                color=c,
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

    out_path = f"/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story/story_consistency_visuals_combined_{dataset_name}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved Combined Consistency Visuals to: {out_path}")

    plt.close()  # Close figure to avoid memory leaks


if __name__ == "__main__":
    csv_shiro_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro"
    csv_ada_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_ada"
    
    # We will iterate over baseline files in csv_shiro
    for file in os.listdir(csv_shiro_dir):
        if file.startswith("per_query_baseline_") and file.endswith(".csv"):
            dataset_name = file.replace("per_query_baseline_", "").replace(".csv", "")
            base_csv = os.path.join(csv_shiro_dir, file)
            shiro_csv = os.path.join(csv_shiro_dir, f"per_query_results_{dataset_name}.csv")
            ada_csv = os.path.join(csv_ada_dir, f"per_query_results_{dataset_name}.csv")

            create_consistency_plot(dataset_name, base_csv, shiro_csv, ada_csv)
