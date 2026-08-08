import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


def generate_plot():
    csv_shiro = (
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro/summary_metrics.csv"
    )
    csv_ada = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_ada/summary_metrics.csv"

    for p in (csv_shiro, csv_ada):
        if not os.path.exists(p):
            print(f"Error: {p} not found. Please run generate_summary_csv.py first.")
            return

    df_shiro = pd.read_csv(csv_shiro)
    df_ada = pd.read_csv(csv_ada)

    # baseline always comes from csv_shiro
    df = df_shiro
    all_datasets = sorted(df["dataset"].unique())

    if len(all_datasets) == 0:
        print("No datasets found in summary_metrics.csv")
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

    # Setup figure layout dynamically to match reference aspect ratio
    ncols = min(num_ds, 4)
    nrows = math.ceil(num_ds / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 5.2 * nrows))

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Hide unused subplots
    for i in range(num_ds, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for idx, ds_name in enumerate(datasets):
        ax = axes_flat[idx]
        ds_df = df[df["dataset"] == ds_name]

        # 1. Baseline Data
        base_df = ds_df[ds_df["method"] == "baseline"].copy()
        base_df["ef"] = pd.to_numeric(base_df["ef"], errors="coerce")
        base_df = base_df.sort_values("ef")

        window_size = 3

        # Smooth BOTH latency (x) and recall (y) using pandas rolling mean with a small window
        x_avg_s = (
            base_df["avg_lat(ns)"]
            .rolling(window=window_size, min_periods=1, center=True)
            .mean()
            / 1e6
        )
        y_avg_s = (
            base_df["avg_recall"]
            .rolling(window=window_size, min_periods=1, center=True)
            .mean()
        )

        x_p05_s = (
            base_df["lat_5th_rec"]
            .rolling(window=window_size, min_periods=1, center=True)
            .mean()
            / 1e6
        )
        y_p05_s = (
            base_df["5th_perc_recall"]
            .rolling(window=window_size, min_periods=1, center=True)
            .mean()
        )

        x_p01_s = (
            base_df["lat_1st_rec"]
            .rolling(window=window_size, min_periods=1, center=True)
            .mean()
            / 1e6
        )
        y_p01_s = (
            base_df["1st_perc_recall"]
            .rolling(window=window_size, min_periods=1, center=True)
            .mean()
        )

        # Draw Lines matching reference style
        # Avg -> Black Dash-Dot
        ax.plot(
            x_avg_s, y_avg_s, "-.", color="black", label="Baseline - Avg", linewidth=1.5
        )
        # p05 -> Red Dashed
        ax.plot(
            x_p05_s,
            y_p05_s,
            "--",
            color="red",
            label=r"Baseline - $\leq\pi_{0.05}$",
            linewidth=1.0,
        )
        # p01 -> Red Dotted
        ax.plot(
            x_p01_s,
            y_p01_s,
            ":",
            color="red",
            label=r"Baseline - $\leq\pi_{0.01}$",
            linewidth=1.5,
        )

        # 2. Shiro-EF adaptive
        shiro_ada = df_shiro[
            (df_shiro["dataset"] == ds_name) & (df_shiro["method"] == "adaptive")
        ]
        if not shiro_ada.empty:
            r = shiro_ada.iloc[0]
            # Avg -> Blue Star
            ax.scatter(
                [r["avg_lat(ns)"] / 1e6],
                [r["avg_recall"]],
                marker="*",
                s=100,
                color="tab:blue",
                zorder=6,
            )
            # p05 -> Blue Diamond
            ax.scatter(
                [r["lat_5th_rec"] / 1e6],
                [r["5th_perc_recall"]],
                marker="D",
                s=40,
                color="tab:blue",
                zorder=5,
            )
            # p01 -> Blue Down-Triangle
            ax.scatter(
                [r["lat_1st_rec"] / 1e6],
                [r["1st_perc_recall"]],
                marker="v",
                s=45,
                color="tab:blue",
                zorder=5,
            )

        # 3. Ada-EF adaptive
        ada_row = df_ada[
            (df_ada["dataset"] == ds_name) & (df_ada["method"] == "adaptive")
        ]
        if not ada_row.empty:
            r = ada_row.iloc[0]
            # Avg -> Orange Circle
            ax.scatter(
                [r["avg_lat(ns)"] / 1e6],
                [r["avg_recall"]],
                marker="o",
                s=45,
                color="tab:orange",
                zorder=5,
            )
            # p05 -> Orange Up-Triangle
            ax.scatter(
                [r["lat_5th_rec"] / 1e6],
                [r["5th_perc_recall"]],
                marker="^",
                s=45,
                color="tab:orange",
                zorder=5,
            )
            # p01 -> Orange Pentagon
            ax.scatter(
                [r["lat_1st_rec"] / 1e6],
                [r["1st_perc_recall"]],
                marker="p",
                s=45,
                color="tab:orange",
                zorder=5,
            )

        # 4. Formatting to match reference image
        ax.set_title(ds_name, fontsize=16)
        ax.set_box_aspect(1)

        ax.set_xlabel("Avg Latency (ms)", fontsize=16)
        # Adapt Y label dynamically or fix to standard
        y_label = (
            "Recall@1000"
            if "1000" in ds_name or "m" in ds_name.lower()
            else "Recall@100"
        )
        ax.set_ylabel(y_label, fontsize=16)

        # Rotate x labels 90 degrees
        ax.tick_params(axis="x", labelrotation=90, labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        ax.set_ylim(bottom=0.7, top=1.02)

        ax.axhline(0.95, color="lightgray", linestyle="--", zorder=0)
        ax.axhline(0.98, color="lightgray", linestyle="--", zorder=0)

        all_times = (
            pd.concat(
                [base_df["avg_lat(ns)"], base_df["lat_5th_rec"], base_df["lat_1st_rec"]]
            )
            / 1e6
        )
        max_time = all_times.max() if not all_times.empty else 10
        for src in (shiro_ada, ada_row):
            if not src.empty:
                r = src.iloc[0]
                max_time = max(
                    max_time,
                    r["avg_lat(ns)"] / 1e6,
                    r["lat_5th_rec"] / 1e6,
                    r["lat_1st_rec"] / 1e6,
                )
        ax.set_xlim(0, max_time * 1.1)

    fig.align_xlabels()
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0, rect=(0, 0.10, 1, 0.94))

    # Unified Legend matching the drawn elements
    legend_handles = [
        Line2D([0], [0], color="black", linestyle="-.", lw=1.5, label="Baseline – Avg"),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            lw=1.0,
            label=r"Baseline – $\leq\pi_{0.05}$",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle=":",
            lw=1.5,
            label=r"Baseline – $\leq\pi_{0.01}$",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="tab:blue",
            markersize=13,
            label="Shiro-EF – Avg",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="tab:blue",
            markersize=7,
            label=r"Shiro-EF – $\leq\pi_{0.05}$",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="tab:blue",
            markersize=9,
            label=r"Shiro-EF – $\leq\pi_{0.01}$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:orange",
            markersize=8,
            label="Ada – Avg",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="tab:orange",
            markersize=9,
            label=r"Ada – $\leq\pi_{0.05}$",
        ),
        Line2D(
            [0],
            [0],
            marker="p",
            color="w",
            markerfacecolor="tab:orange",
            markersize=8,
            label=r"Ada – $\leq\pi_{0.01}$",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=9,  # Flat arrangement
        fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.08),
        columnspacing=0.8,
        handletextpad=0.3,
    )

    os.makedirs("/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img", exist_ok=True)
    out_path = (
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/visualization_final.png"
    )
    plt.savefig(out_path, dpi=300)
    print(f"Saved styled plot to {out_path}")


if __name__ == "__main__":
    generate_plot()
