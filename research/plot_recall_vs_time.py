"""
Plot per-query recall vs. total search time.

Shiro-EF: Plotted as a Box-and-Whisker at its specific total search time.
Ada-EF: Plotted as a Box-and-Whisker at its specific total search time.
Baseline: Plotted as multiple Box-and-Whisker plots (one for each static EF level),
          with a line connecting their means.

Usage:
    python3 research/plot_recall_vs_time.py
    python3 research/plot_recall_vs_time.py --dataset deep-image-96-angular
"""

import argparse
import glob
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_DIR = os.path.join(SCRIPT_DIR, "csv_shiro")
DEFAULT_ADA_DIR = os.path.join(SCRIPT_DIR, "csv_ada")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "img", "recall_vs_time")

SHIRO_COLOR    = "#2563eb"   # blue
ADA_COLOR      = "#f59e0b"   # amber/orange
BASELINE_COLOR = "#dc2626"   # red
MEAN_REF_COLOR = "#16a34a"   # green

WHIS = (5, 95)    # whisker percentiles

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_shiro(csv_dir: str, dataset: str) -> pd.DataFrame | None:
    path = os.path.join(csv_dir, f"per_query_results_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df

def load_ada(ada_dir: str, dataset: str) -> pd.DataFrame | None:
    path = os.path.join(ada_dir, f"per_query_results_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df

def load_baseline(csv_dir: str, dataset: str) -> pd.DataFrame | None:
    path = os.path.join(csv_dir, f"per_query_baseline_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df

def total_time_s(df: pd.DataFrame) -> float:
    return float(df["Latency(ns)"].sum()) / 1e9

def box_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean":  float(np.mean(arr)),
        "med":   float(np.median(arr)),
        "q1":    float(np.percentile(arr, 25)),
        "q3":    float(np.percentile(arr, 75)),
        "w_lo":  float(np.percentile(arr, WHIS[0])),
        "w_hi":  float(np.percentile(arr, WHIS[1])),
    }

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_box(ax, x: float, stats: dict, color: str, width: float,
             zorder: int = 3, alpha: float = 0.85, show_mean: bool = True):
    q1, q3   = stats["q1"],   stats["q3"]
    w_lo, w_hi = stats["w_lo"], stats["w_hi"]
    med      = stats["med"]
    mean_val = stats["mean"]

    # Whiskers
    ax.plot([x, x], [w_lo, q1],  color=color, lw=1.4, zorder=zorder, alpha=alpha)
    ax.plot([x, x], [q3, w_hi],  color=color, lw=1.4, zorder=zorder, alpha=alpha)
    for y in (w_lo, w_hi):
        ax.plot([x - width / 2, x + width / 2], [y, y],
                color=color, lw=1.4, zorder=zorder, alpha=alpha)

    # Box
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, q1), width, q3 - q1,
        boxstyle="square,pad=0",
        linewidth=1.4, edgecolor=color,
        facecolor=color, alpha=alpha * 0.35,
        zorder=zorder,
    )
    ax.add_patch(rect)

    # Median line
    ax.plot([x - width / 2, x + width / 2], [med, med],
            color=color, lw=2.0, zorder=zorder + 1)

    # Mean diamond
    if show_mean:
        ax.scatter([x], [mean_val], marker="D", s=40, color=color,
                   edgecolors="white", linewidths=0.8, zorder=zorder + 2)

# ---------------------------------------------------------------------------
# Main plotting logic
# ---------------------------------------------------------------------------

def plot_dataset(dataset: str, csv_dir: str, ada_dir: str, out_dir: str):
    shiro_df = load_shiro(csv_dir, dataset)
    ada_df   = load_ada(ada_dir, dataset)
    base_df  = load_baseline(csv_dir, dataset)

    if shiro_df is None and base_df is None and ada_df is None:
        print(f"[{dataset}] No data found in {csv_dir} or {ada_dir}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")

    legend_handles = []
    
    # 1. First pass to find all X coordinates for box width scaling
    all_x = []
    if base_df is not None and not base_df.empty:
        ef_levels = sorted(base_df["EF"].unique())
        for ef in ef_levels:
            subset = base_df[base_df["EF"] == ef]
            all_x.append(total_time_s(subset))
            
    if shiro_df is not None:
        all_x.append(total_time_s(shiro_df))

    if ada_df is not None:
        all_x.append(total_time_s(ada_df))

    # Calculate dynamic box width
    if len(all_x) > 1 and max(all_x) > min(all_x):
        span = max(all_x) - min(all_x)
        box_width = span * 0.02
    else:
        box_width = all_x[0] * 0.05 if all_x else 0.05
        span = box_width * 10

    # 2. Plot Baseline as multiple Box-and-Whisker plots + Connecting Line
    if base_df is not None and not base_df.empty:
        ef_levels = sorted(base_df["EF"].unique())
        
        base_x = []
        base_y = []
        
        for ef in ef_levels:
            subset = base_df[base_df["EF"] == ef]
            x_val = total_time_s(subset)
            stats = box_stats(subset["Recall"].to_numpy())
            
            base_x.append(x_val)
            base_y.append(stats["mean"])
            
            # Draw baseline box without the red mean marker dot/diamond
            draw_box(ax, x_val, stats, BASELINE_COLOR, width=box_width, alpha=0.7, show_mean=False)
                        
        # Draw the line connecting the means
        ax.plot(base_x, base_y, color=BASELINE_COLOR, lw=1.5, ls='-', zorder=2, alpha=0.6)
                
        legend_handles.append(mpatches.Patch(
            facecolor=BASELINE_COLOR, alpha=0.6,
            label=f"Baseline HNSW ({len(ef_levels)} EF levels)"
        ))

    # 3. Plot Ada-EF as Box-and-Whisker
    if ada_df is not None:
        x_ada = total_time_s(ada_df)
        stats = box_stats(ada_df["Recall"].to_numpy())
        
        draw_box(ax, x_ada, stats, ADA_COLOR, width=box_width)

        # Vertical reference line
        ax.axvline(x_ada, color=ADA_COLOR, lw=1.0, ls=":", alpha=0.5)
        
        legend_handles.append(mpatches.Patch(
            facecolor=ADA_COLOR, alpha=0.6,
            label=f"Ada-EF (Total Time: {x_ada:.2f}s)"
        ))

    # 4. Plot Shiro-EF as Box-and-Whisker
    if shiro_df is not None:
        x_shiro = total_time_s(shiro_df)
        stats = box_stats(shiro_df["Recall"].to_numpy())
        
        draw_box(ax, x_shiro, stats, SHIRO_COLOR, width=box_width)

        ax.axhline(stats["mean"],
                   color=MEAN_REF_COLOR, lw=1.5, ls="--", alpha=0.7)

        # Vertical reference line
        ax.axvline(x_shiro, color=SHIRO_COLOR, lw=1.0, ls=":", alpha=0.5)
        
        legend_handles.append(mpatches.Patch(
            facecolor=SHIRO_COLOR, alpha=0.6,
            label=f"Shiro-EF (Total Time: {x_shiro:.2f}s)"
        ))

    # Add standard legend elements
    legend_handles += [
        plt.scatter([], [], marker="D", s=40, color="gray",
                    edgecolors="white", linewidths=0.8, label="Mean Recall"),
        Line2D([0], [0], color="gray", lw=2.0, label="Median Recall"),
        Line2D([0], [0], color=MEAN_REF_COLOR, lw=1.5, ls="--",
               label="Shiro-EF Mean Recall Reference"),
    ]

    # Axes formatting
    margin = span * 0.1
    if all_x:
        ax.set_xlim(max(0, min(all_x) - margin), max(all_x) + margin)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))

    ax.set_xlabel("Total Search Time (s)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Per-Query Recall", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Recall Distribution vs. Total Search Time\n{dataset}",
        fontsize=14, fontweight="bold", pad=12,
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(axis="y", color="#cbd5e1", lw=0.8, alpha=0.7)
    ax.grid(axis="x", color="#cbd5e1", lw=0.5, alpha=0.4, ls=":")

    note = f"Box: IQR | Whiskers: {WHIS[0]}th–{WHIS[1]}th pct"
    ax.text(0.99, 0.01, note, transform=ax.transAxes,
            fontsize=8, color="#64748b", ha="right", va="bottom")

    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#cbd5e1")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"recall_vs_time_{dataset}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def discover_datasets(csv_dir: str, ada_dir: str) -> list[str]:
    # Check both directories for datasets
    paths_shiro = glob.glob(os.path.join(csv_dir, "per_query_results_*.csv"))
    paths_ada   = glob.glob(os.path.join(ada_dir, "per_query_results_*.csv"))
    
    datasets = set()
    for p in paths_shiro + paths_ada:
        ds = os.path.basename(p).removeprefix("per_query_results_").removesuffix(".csv")
        datasets.add(ds)
        
    return sorted(list(datasets))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None,
                        help="Dataset name (default: all discovered datasets)")
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR,
                        help=f"CSV directory for Shiro/Baseline (default: {DEFAULT_CSV_DIR})")
    parser.add_argument("--ada-dir", default=DEFAULT_ADA_DIR,
                        help=f"CSV directory for Ada-EF (default: {DEFAULT_ADA_DIR})")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help=f"Output image directory (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else discover_datasets(args.csv_dir, args.ada_dir)
    if not datasets:
        print(f"No per_query_results_*.csv files found in {args.csv_dir} or {args.ada_dir}")
        sys.exit(1)

    for ds in datasets:
        plot_dataset(ds, args.csv_dir, args.ada_dir, args.out)


if __name__ == "__main__":
    main()
