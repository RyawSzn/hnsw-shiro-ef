import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

CSV_SHIRO = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro"
CSV_ADA = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_ada"
IMG_DIR = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story"

def adaptive_noise_tax(df_base, df_results):
    merged = df_base.merge(
        df_results[["QueryID", "Recall", "Latency(ns)"]],
        on="QueryID",
        suffixes=("_base", "_mine"),
    )
    merged.sort_values(["QueryID", "EF"], inplace=True)

    def min_ef_latency(grp):
        suff = grp[grp["Recall_base"] >= grp["Recall_mine"]]
        return (
            suff.iloc[0]["Latency(ns)_base"]
            if not suff.empty
            else grp.iloc[-1]["Latency(ns)_base"]
        )

    essential_ms = merged.groupby("QueryID").apply(min_ef_latency).sum() / 1e6
    total_ms = df_results["Latency(ns)"].sum() / 1e6
    return max(0.0, total_ms - essential_ms), df_results["Recall"].mean()

def generate_plot():
    os.makedirs(IMG_DIR, exist_ok=True)

    tax_path = os.path.join(CSV_SHIRO, "noise_tax_table_all_efs.csv")
    if not os.path.exists(tax_path):
        print("Run calculate_noise_tax_table.py first.")
        return

    df_tax = pd.read_csv(tax_path)
    
    all_datasets = sorted(df_tax["Dataset"].unique())

    if len(all_datasets) == 0:
        print("No datasets found in noise_tax_table_all_efs.csv")
        return

    print("Available datasets:")
    for i, ds in enumerate(all_datasets):
        print(f"  [{i}] {ds}")

    user_input = input("\nEnter dataset numbers to plot (comma-separated), or press Enter for all: ").strip()

    if not user_input or user_input.lower() == 'all':
        datasets = all_datasets
    else:
        selected_indices = []
        for p in user_input.split(','):
            p = p.strip()
            if p.isdigit():
                idx = int(p)
                if 0 <= idx < len(all_datasets):
                    selected_indices.append(idx)
                else:
                    print(f"Warning: Index {idx} out of bounds, skipping.")
            else:
                print(f"Warning: Invalid input '{p}', skipping.")
        
        # Preserve selection order and remove duplicates
        datasets = [all_datasets[i] for i in dict.fromkeys(selected_indices)]

    num_ds = len(datasets)
    if num_ds == 0:
        print("No datasets selected. Exiting.")
        return

    # Setup figure layout dynamically to match plot_final.py aspect ratio
    ncols = min(num_ds, 4)
    nrows = math.ceil(num_ds / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 5.2 * nrows))
    fig.patch.set_facecolor("white")

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Hide unused subplots
    for i in range(num_ds, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        
        base_csv = os.path.join(CSV_SHIRO, f"per_query_baseline_{dataset}.csv")
        shiro_mine_csv = os.path.join(CSV_SHIRO, f"per_query_results_{dataset}.csv")
        ada_mine_csv = os.path.join(CSV_ADA, f"per_query_results_{dataset}.csv")

        if not os.path.exists(base_csv) or not os.path.exists(shiro_mine_csv):
            print(f"Missing base or shiro csv for {dataset}")
            continue

        df_base = pd.read_csv(base_csv)
        df_shiro = pd.read_csv(shiro_mine_csv)

        base_recalls = df_base.groupby("EF")["Recall"].mean().reset_index()
        base_recalls.rename(columns={"EF": "Global EF (ef)"}, inplace=True)

        df_ds = df_tax[df_tax["Dataset"] == dataset].copy()
        df_ds = df_ds.merge(base_recalls, on="Global EF (ef)")
        df_ds.sort_values("Recall", inplace=True)

        shiro_tax_ms, shiro_recall = adaptive_noise_tax(df_base, df_shiro)

        has_ada = os.path.exists(ada_mine_csv)
        if has_ada:
            df_ada = pd.read_csv(ada_mine_csv)
            ada_tax_ms, ada_recall = adaptive_noise_tax(df_base, df_ada)

        # Plot Baseline
        ax.plot(
            df_ds["Recall"],
            df_ds["Noise Tax (ms)"],
            linewidth=2.5,
            color="#e74c3c",
            label="Static EF (Baseline)",
        )



        # Plot Shiro-EF
        ax.scatter(
            [shiro_recall],
            [shiro_tax_ms],
            color="tab:blue",
            marker="*",
            s=100,
            zorder=5,
            label="SHIRO-EF",
        )
        ax.axvline(x=shiro_recall, color="tab:blue", linestyle=":", alpha=0.5)
        ax.axhline(y=shiro_tax_ms, color="tab:blue", linestyle=":", alpha=0.5)

        # Plot Ada-EF
        if has_ada:
            ax.scatter(
                [ada_recall],
                [ada_tax_ms],
                color="tab:orange",
                marker="o",
                s=50,
                zorder=5,
                label="Ada-EF",
            )
            ax.axvline(x=ada_recall, color="tab:orange", linestyle=":", alpha=0.5)
            ax.axhline(y=ada_tax_ms, color="tab:orange", linestyle=":", alpha=0.5)

        # Formatting
        ax.set_title(dataset, fontsize=16)
        ax.set_box_aspect(1)

        ax.set_xlabel("Mean Recall", fontsize=16)
        ax.set_ylabel("Noise Tax (ms)", fontsize=16)

        ax.tick_params(axis="x", labelrotation=90, labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        
        

    fig.align_xlabels()
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0, rect=(0, 0.10, 1, 0.94))

    # Unified Legend
    legend_handles = [
        Line2D([0], [0], color="#e74c3c", lw=2.5, label="Static EF (Baseline)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="tab:blue", markersize=10, label="Shiro-EF"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=7, label="Ada-EF"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.08),
        columnspacing=1.5,
        handletextpad=0.5,
    )

    out_path = os.path.join(IMG_DIR, "noise_tax_vs_recall_all.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved styled plot to {out_path}")

if __name__ == "__main__":
    generate_plot()
