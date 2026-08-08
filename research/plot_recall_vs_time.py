"""
Plot per-query recall vs. total search time using Dodged Box-and-Whisker plots.

Shiro-EF & Ada-EF: Box plots visually shifted (dodged) left/right to prevent overlap,
                   with vertical lines marking their true search time.
Baseline: Box plots placed at true search times.
Mean Line: Heavily thickened to emphasize the average performance trend.

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
    return None if df.empty else df

def load_ada(ada_dir: str, dataset: str) -> pd.DataFrame | None:
    path = os.path.join(ada_dir, f"per_query_results_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return None if df.empty else df

def load_baseline(csv_dir: str, dataset: str) -> pd.DataFrame | None:
    path = os.path.join(csv_dir, f"per_query_baseline_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return None if df.empty else df

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

def draw_box(ax, true_x: float, draw_x: float, stats: dict, color: str, width: float,
             zorder: int = 3, alpha: float = 0.85, show_mean: bool = True):
    q1, q3   = stats["q1"],   stats["q3"]
    w_lo, w_hi = stats["w_lo"], stats["w_hi"]
    med      = stats["med"]
    mean_val = stats["mean"]

    # Thin Whiskers (drawn at visual shifted position)
    ax.plot([draw_x, draw_x], [w_lo, q1], color=color, lw=1.5, alpha=alpha, zorder=zorder)
    ax.plot([draw_x, draw_x], [q3, w_hi], color=color, lw=1.5, alpha=alpha, zorder=zorder)
    
    # Modern Box
    rect = mpatches.Rectangle((draw_x - width / 2, q1), width, q3 - q1,
                              facecolor=color, edgecolor="none", alpha=alpha * 0.4, zorder=zorder)
    ax.add_patch(rect)
    ax.plot([draw_x - width/2, draw_x - width/2], [q1, q3], color=color, lw=2.0, alpha=alpha, zorder=zorder)
    ax.plot([draw_x + width/2, draw_x + width/2], [q1, q3], color=color, lw=2.0, alpha=alpha, zorder=zorder)

    # Thick Median line
    ax.plot([draw_x - width / 2, draw_x + width / 2], [med, med],
            color=color, lw=2.5, zorder=zorder + 1, alpha=min(1.0, alpha+0.2))

    # Optional linker line if dodged
    if true_x != draw_x:
        ax.plot([draw_x, true_x], [mean_val, mean_val], color=color, ls=":", lw=1.5, alpha=0.5, zorder=zorder-1)

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

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")

    legend_handles = []
    
    all_x = []
    if base_df is not None and not base_df.empty:
        for ef in base_df["EF"].unique():
            all_x.append(total_time_s(base_df[base_df["EF"] == ef]))
            
    if shiro_df is not None:
        all_x.append(total_time_s(shiro_df))
    if ada_df is not None:
        all_x.append(total_time_s(ada_df))

    # Standard visible box width
    if len(all_x) > 1 and max(all_x) > min(all_x):
        span = max(all_x) - min(all_x)
        box_width = span * 0.022 
    else:
        box_width = all_x[0] * 0.05 if all_x else 0.05
        span = box_width * 10

    # 1. Plot Baseline as multiple Box plots + STRONG Connecting Mean Line
    if base_df is not None and not base_df.empty:
        ef_levels = sorted(base_df["EF"].unique())
        base_x, base_y = [], []
        
        for ef in ef_levels:
            subset = base_df[base_df["EF"] == ef]
            x_val = total_time_s(subset)
            stats = box_stats(subset["Recall"].to_numpy())
            
            base_x.append(x_val)
            base_y.append(stats["mean"])
            
            # Baseline is drawn directly at true X
            draw_box(ax, true_x=x_val, draw_x=x_val, stats=stats, color=BASELINE_COLOR, width=box_width, zorder=2, alpha=0.5, show_mean=False)
                        
        # STRONGER connecting line for Baseline Means
        ax.plot(base_x, base_y, color=BASELINE_COLOR, lw=4.5, ls='-', zorder=10, alpha=0.9)
                
        legend_handles.append(mpatches.Patch(
            facecolor=BASELINE_COLOR, alpha=0.6,
            label=f"Baseline HNSW ({len(ef_levels)} EF levels)"
        ))

    # 2. Plot Ada-EF as Box-and-Whisker (Dodged Left)
    if ada_df is not None:
        x_ada = total_time_s(ada_df)
        stats = box_stats(ada_df["Recall"].to_numpy())
        
        # Shift visually to the left by 0.8 * box_width to prevent overlap
        draw_x = x_ada - box_width * 0.8
        
        draw_box(ax, true_x=x_ada, draw_x=draw_x, stats=stats, color=ADA_COLOR, width=box_width, zorder=4, alpha=0.9, show_mean=False)
        ax.axvline(x_ada, color=ADA_COLOR, lw=1.5, ls=":", alpha=0.7, zorder=1)
        
        # Stronger Ada mean marker
        ax.scatter([draw_x], [stats["mean"]], marker="D", s=110, color=ADA_COLOR, edgecolors="white", linewidths=1.5, zorder=12)
        
        legend_handles.append(mpatches.Patch(
            facecolor=ADA_COLOR, alpha=0.9,
            label=f"Ada-EF (Time: {x_ada:.2f}s)"
        ))

    # 3. Plot Shiro-EF as Box-and-Whisker (Dodged Right)
    if shiro_df is not None:
        x_shiro = total_time_s(shiro_df)
        stats = box_stats(shiro_df["Recall"].to_numpy())
        
        # Shift visually to the right by 0.8 * box_width to prevent overlap
        draw_x = x_shiro + box_width * 0.8
        
        draw_box(ax, true_x=x_shiro, draw_x=draw_x, stats=stats, color=SHIRO_COLOR, width=box_width, zorder=5, alpha=0.9, show_mean=False)
        
        # STRONGER Shiro mean horizontal reference line
        ax.axhline(stats["mean"], color=MEAN_REF_COLOR, lw=3.0, ls="--", alpha=0.9, zorder=1)
        ax.axvline(x_shiro, color=SHIRO_COLOR, lw=1.5, ls=":", alpha=0.7, zorder=1)
        
        # Stronger Shiro mean marker
        ax.scatter([draw_x], [stats["mean"]], marker="D", s=110, color=SHIRO_COLOR, edgecolors="white", linewidths=1.5, zorder=12)
        
        legend_handles.append(mpatches.Patch(
            facecolor=SHIRO_COLOR, alpha=0.9,
            label=f"Shiro-EF (Time: {x_shiro:.2f}s)"
        ))

    # Add standard legend elements
    legend_handles += [
        Line2D([0], [0], color="gray", lw=4.5, ls="-", label="Mean Recall Trend"),
        Line2D([0], [0], color=MEAN_REF_COLOR, lw=3.0, ls="--",
               label="Shiro-EF Mean Recall Reference"),
    ]

    # Axes formatting
    margin = span * 0.15  # Slightly larger margin to accommodate dodging
    if all_x:
        ax.set_xlim(max(0, min(all_x) - margin), max(all_x) + margin)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))

    ax.set_xlabel("Total Search Time (s)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Per-Query Recall", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Recall Distribution vs. Total Search Time (Dodged)\n{dataset}",
        fontsize=14, fontweight="bold", pad=12,
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(axis="y", color="#cbd5e1", lw=0.8, alpha=0.7, zorder=0)
    ax.grid(axis="x", color="#cbd5e1", lw=0.5, alpha=0.4, ls=":", zorder=0)

    note = f"Box: IQR | Whiskers: {WHIS[0]}th–{WHIS[1]}th pct | Means emphasized | Boxes Dodged Left/Right"
    ax.text(0.99, 0.01, note, transform=ax.transAxes,
            fontsize=8, color="#64748b", ha="right", va="bottom")

    ax.legend(handles=legend_handles, loc="lower right", fontsize=9.5,
              framealpha=0.95, edgecolor="#cbd5e1")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"recall_vs_time_{dataset}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def discover_datasets(csv_dir: str, ada_dir: str) -> list[str]:
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
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    parser.add_argument("--ada-dir", default=DEFAULT_ADA_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else discover_datasets(args.csv_dir, args.ada_dir)
    if not datasets:
        print(f"No per_query_results_*.csv files found in {args.csv_dir} or {args.ada_dir}")
        sys.exit(1)

    for ds in datasets:
        plot_dataset(ds, args.csv_dir, args.ada_dir, args.out)


if __name__ == "__main__":
    main()
