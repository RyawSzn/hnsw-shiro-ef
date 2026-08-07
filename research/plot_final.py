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
    datasets = sorted(df["dataset"].unique())
    num_ds = len(datasets)

    if num_ds == 0:
        print("No datasets found in summary_metrics.csv")
        return

    # Setup figure layout dynamically to avoid empty subplots
    ncols = min(num_ds, 3)
    nrows = math.ceil(num_ds / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 8.5 * nrows))

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

        # Sort by EF to smooth along the parameter sweep
        base_df["ef"] = pd.to_numeric(base_df["ef"], errors="coerce")
        base_df = base_df.sort_values("ef")

        # REDUCED SMOOTHING: Very small window size
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

        ax.plot(
            x_avg_s,
            y_avg_s,
            "-",
            color="tab:blue",
            label="Baseline - Avg",
            linewidth=2.5,
        )
        ax.plot(
            x_p05_s,
            y_p05_s,
            "-",
            color="tab:green",
            label=r"Baseline - $<\pi_{0.05}$",
            linewidth=2.5,
        )
        ax.plot(
            x_p01_s,
            y_p01_s,
            "-",
            color="tab:orange",
            label=r"Baseline - $<\pi_{0.01}$",
            linewidth=2.5,
        )

        # 2a. shiro-ef adaptive → star (★)
        shiro_ada = df_shiro[
            (df_shiro["dataset"] == ds_name) & (df_shiro["method"] == "adaptive")
        ]
        if not shiro_ada.empty:
            r = shiro_ada.iloc[0]
            ax.scatter(
                [r["avg_lat(ns)"] / 1e6],
                [r["avg_recall"]],
                marker="*",
                s=350,
                color="blue",
                edgecolor="black",
                zorder=5,
            )
            ax.scatter(
                [r["lat_5th_rec"] / 1e6],
                [r["5th_perc_recall"]],
                marker="*",
                s=350,
                color="green",
                edgecolor="black",
                zorder=5,
            )
            ax.scatter(
                [r["lat_1st_rec"] / 1e6],
                [r["1st_perc_recall"]],
                marker="*",
                s=350,
                color="orange",
                edgecolor="black",
                zorder=5,
            )

        # 2b. ada adaptive → cross (x)
        ada_row = df_ada[
            (df_ada["dataset"] == ds_name) & (df_ada["method"] == "adaptive")
        ]
        if not ada_row.empty:
            r = ada_row.iloc[0]
            ax.scatter(
                [r["avg_lat(ns)"] / 1e6],
                [r["avg_recall"]],
                marker="x",
                s=180,
                color="blue",
                linewidths=2.5,
                zorder=5,
            )
            ax.scatter(
                [r["lat_5th_rec"] / 1e6],
                [r["5th_perc_recall"]],
                marker="x",
                s=180,
                color="green",
                linewidths=2.5,
                zorder=5,
            )
            ax.scatter(
                [r["lat_1st_rec"] / 1e6],
                [r["1st_perc_recall"]],
                marker="x",
                s=180,
                color="orange",
                linewidths=2.5,
                zorder=5,
            )

        # 3. Formatting
        max_ef = pd.to_numeric(base_df["ef"], errors="coerce").max()
        ax.set_title(
            f"{ds_name}\nObserved ef_max = {int(max_ef) if pd.notna(max_ef) else 'Unknown'}",
            fontsize=14,
        )

        ax.set_xlabel("Avg Latency (ms)", fontsize=12)
        ax.set_ylabel("Recall@100", fontsize=12)

        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_ylim(0.70, 1.01)

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

        # ADD TARGET RECALL LINE
        ax.axhline(
            y=0.95, color="tab:red", linestyle="-.", alpha=0.8, linewidth=1.5, zorder=1
        )

        # ADD TEXT STATING IT'S THE TARGET RECALL
        x_min, x_max = ax.get_xlim()
        offset = (x_max - x_min) * 0.02
        ax.text(
            x_min + offset,
            0.972,
            "Target Recall",
            color="tab:red",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=6,
        )
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=(0, 0.08, 1, 1))

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=2.5, label="Baseline – Avg"),
        Line2D([0], [0], color="tab:green", lw=2.5, label=r"Baseline – $<\pi_{0.05}$"),
        Line2D([0], [0], color="tab:orange", lw=2.5, label=r"Baseline – $<\pi_{0.01}$"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="blue",
            markersize=13,
            markeredgecolor="black",
            label="Shiro-EF – Avg",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="green",
            markersize=13,
            markeredgecolor="black",
            label=r"Shiro-EF – $<\pi_{0.05}$",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="orange",
            markersize=13,
            markeredgecolor="black",
            label=r"Shiro-EF – $<\pi_{0.01}$",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="blue",
            markersize=10,
            linewidth=0,
            markeredgewidth=2.5,
            label="Ada – Avg",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="green",
            markersize=10,
            linewidth=0,
            markeredgewidth=2.5,
            label=r"Ada – $<\pi_{0.05}$",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="orange",
            markersize=10,
            linewidth=0,
            markeredgewidth=2.5,
            label=r"Ada – $<\pi_{0.01}$",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=10,
        framealpha=0.95,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.01),
    )

    os.makedirs("/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img", exist_ok=True)
    out_path = (
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/visualization_final.png"
    )
    plt.savefig(out_path, dpi=300)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    generate_plot()
