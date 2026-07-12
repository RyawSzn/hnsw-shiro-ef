#!/usr/bin/env python3
"""
plot_ef_sweep.py — Parse ef_sweep log and plot 3-D latency vs recall vs ef_max
                   for each dataset.

Usage:
    python3 research/plot_ef_sweep.py research/ef_sweep.log

Output:
    research/ef_sweep_3d.png   — one 3-D subplot per dataset
    research/ef_sweep_2d.png   — companion 2-D recall & latency curves
"""

import sys
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3-D projection)
from matplotlib import cm

# ---------------------------------------------------------------------------
# 1. Parse the log
# ---------------------------------------------------------------------------

def parse_log(path: str) -> tuple[dict, dict]:
    """
    Returns:
      data: { dataset: { "ef": [...], "recall": [...], "latency_ms": [...] } }
      caps: { dataset: natural_cap_int }
    Only RESULT| and CAP| lines are consumed.
    """
    data: dict[str, dict] = defaultdict(lambda: {"ef": [], "recall": [], "latency_ms": []})
    caps: dict[str, int]  = {}

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("CAP|"):
                parts = line.split("|")
                if len(parts) == 3:
                    caps[parts[1]] = int(parts[2])
            elif line.startswith("RESULT|"):
                parts = line.split("|")
                if len(parts) != 5:
                    continue
                _, dataset, ef_str, recall_str, latency_str = parts
                data[dataset]["ef"].append(int(ef_str))
                data[dataset]["recall"].append(float(recall_str))
                data[dataset]["latency_ms"].append(float(latency_str))

    return dict(data), caps


# ---------------------------------------------------------------------------
# 2. Find "best" ef_max
#    Strategy: highest recall that's still within 10 % of the minimum latency
#    (knee-point in the latency/recall trade-off).
# ---------------------------------------------------------------------------

def find_best_ef(ef_vals, recall_vals, latency_vals, recall_threshold: float = 0.95):
    """
    Knee-point on the recall-vs-latency curve: the ef_max where you get the
    most recall per unit of added latency.

    Method: normalize both axes to [0,1], then find the point with maximum
    perpendicular distance from the line connecting the first and last point
    (the standard "elbow" / knee detection algorithm).

    Falls back to lowest-latency point with recall >= recall_threshold if the
    knee lands below the threshold, and to highest-recall if nothing reaches it.
    """
    arr_rec = np.array(recall_vals, dtype=float)
    arr_lat = np.array(latency_vals, dtype=float)

    rec_n = (arr_rec - arr_rec.min()) / (arr_rec.max() - arr_rec.min() + 1e-12)
    lat_n = (arr_lat - arr_lat.min()) / (arr_lat.max() - arr_lat.min() + 1e-12)

    # vector from first to last point in normalized space
    p1 = np.array([lat_n[0],  rec_n[0]])
    p2 = np.array([lat_n[-1], rec_n[-1]])
    d  = p2 - p1
    d_norm = np.linalg.norm(d)

    distances = []
    for i in range(len(arr_rec)):
        p = np.array([lat_n[i], rec_n[i]])
        if d_norm < 1e-12:
            distances.append(0.0)
        else:
            # perpendicular distance from point p to line p1->p2 in 2-D
            distances.append(abs((p2[0]-p1[0])*(p1[1]-p[1]) - (p1[0]-p[0])*(p2[1]-p1[1])) / d_norm)

    knee_idx = int(np.argmax(distances))

    if arr_rec[knee_idx] >= recall_threshold:
        return knee_idx

    mask = arr_rec >= recall_threshold
    if mask.any():
        candidates = np.where(mask)[0]
        return int(candidates[np.argmin(arr_lat[candidates])])
    return int(np.argmax(arr_rec))


# ---------------------------------------------------------------------------
# 3. 3-D plot
# ---------------------------------------------------------------------------

NICE_NAMES = {
    "deep-image-96-angular": "Deep-Image-96",
    "glove-100-angular":     "GloVe-100",
    "sift-128-euclidean":    "SIFT-128",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def plot_3d(data: dict, caps: dict, out_path: str):
    n = len(data)
    fig = plt.figure(figsize=(7 * n, 6))
    fig.suptitle("EF Sweep: Latency vs Recall vs EF (per dataset)", fontsize=15, fontweight="bold")

    for col, (dataset, vals) in enumerate(data.items()):
        ax = fig.add_subplot(1, n, col + 1, projection="3d")

        ef      = np.array(vals["ef"])
        recall  = np.array(vals["recall"])
        latency = np.array(vals["latency_ms"])

        # Colour the line by recall value (cool → warm)
        norm   = mcolors.Normalize(recall.min(), recall.max())
        colors = plt.colormaps["viridis"](norm(recall))

        # Draw segment-by-segment with colour
        for i in range(len(ef) - 1):
            ax.plot(
                ef[i:i+2], recall[i:i+2], latency[i:i+2],
                color=colors[i], linewidth=2.0
            )

        # Mark best point
        best_i = find_best_ef(ef, recall, latency)
        ax.scatter(
            ef[best_i], recall[best_i], latency[best_i],
            color="red", s=120, zorder=5,
            label=f"Best ef={ef[best_i]}\nRecall={recall[best_i]:.4f}\nLatency={latency[best_i]:.0f}ms"
        )

        cap = caps.get(dataset)
        cap_label = f"\n(natural cap ef={cap})" if cap is not None else ""
        ax.set_title(NICE_NAMES.get(dataset, dataset) + cap_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("ef_max",    fontsize=10)
        ax.set_ylabel("Avg Recall",fontsize=10)
        ax.set_zlabel("Latency (ms)", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)

        # Colour-bar for recall
        sm = cm.ScalarMappable(cmap=plt.colormaps["viridis"], norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label="Recall")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved 3-D plot → {out_path}")


# ---------------------------------------------------------------------------
# 4. 2-D companion plot (easier to read exact numbers)
# ---------------------------------------------------------------------------

def plot_2d(data: dict, caps: dict, out_path: str):
    n = len(data)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("EF Sweep: Recall & Latency curves", fontsize=14, fontweight="bold")

    for row, (dataset, vals) in enumerate(data.items()):
        ef      = np.array(vals["ef"])
        recall  = np.array(vals["recall"])
        latency = np.array(vals["latency_ms"])
        best_i  = find_best_ef(ef, recall, latency)

        title = NICE_NAMES.get(dataset, dataset)

        # --- Recall curve ---
        ax_r = axes[row][0]
        ax_r.plot(ef, recall, color=COLORS[row % len(COLORS)], linewidth=2)
        ax_r.axvline(ef[best_i], color="red", linestyle="--", linewidth=1.2)
        ax_r.scatter([ef[best_i]], [recall[best_i]], color="red", s=80, zorder=5)
        ax_r.annotate(
            f"ef={ef[best_i]}\nr={recall[best_i]:.4f}",
            xy=(ef[best_i], recall[best_i]),
            xytext=(ef[best_i] + 80, recall[best_i] - 0.015),
            fontsize=8, color="red",
            arrowprops=dict(arrowstyle="->", color="red")
        )
        ax_r.set_title(f"{title} — Recall", fontsize=11)
        ax_r.set_xlabel("ef_max")
        ax_r.set_ylabel("Avg Recall")
        cap = caps.get(dataset)
        if cap is not None:
            ax_r.axvline(cap, color="gray", linestyle=":", linewidth=1.0, label=f"natural cap={cap}")
            ax_r.legend(fontsize=7)
        ax_r.grid(True, alpha=0.3)

        # --- Latency curve ---
        ax_l = axes[row][1]
        ax_l.plot(ef, latency, color=COLORS[row % len(COLORS)], linewidth=2)
        ax_l.axvline(ef[best_i], color="red", linestyle="--", linewidth=1.2)
        ax_l.scatter([ef[best_i]], [latency[best_i]], color="red", s=80, zorder=5)
        ax_l.annotate(
            f"ef={ef[best_i]}\n{latency[best_i]:.0f}ms",
            xy=(ef[best_i], latency[best_i]),
            xytext=(ef[best_i] + 80, latency[best_i] + latency.max() * 0.03),
            fontsize=8, color="red",
            arrowprops=dict(arrowstyle="->", color="red")
        )
        ax_l.set_title(f"{title} — Latency", fontsize=11)
        ax_l.set_xlabel("ef_max")
        ax_l.set_ylabel("Latency (ms)")
        if cap is not None:
            ax_l.axvline(cap, color="gray", linestyle=":", linewidth=1.0, label=f"natural cap={cap}")
            ax_l.legend(fontsize=7)
        ax_l.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved 2-D plot → {out_path}")


# ---------------------------------------------------------------------------
# 5. Summary table
# ---------------------------------------------------------------------------

def print_summary(data: dict, caps: dict):
    print("\n" + "=" * 78)
    print(f"{'Dataset':<30}  {'Cap':>6}  {'Best ef':>8}  {'Recall':>8}  {'Latency(ms)':>12}")
    print("=" * 78)
    for dataset, vals in data.items():
        ef      = np.array(vals["ef"])
        recall  = np.array(vals["recall"])
        latency = np.array(vals["latency_ms"])
        bi      = find_best_ef(ef, recall, latency)
        name    = NICE_NAMES.get(dataset, dataset)
        cap     = caps.get(dataset, "-")
        print(f"{name:<30}  {str(cap):>6}  {ef[bi]:>8d}  {recall[bi]:>8.4f}  {latency[bi]:>12.1f}")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        log_path = os.path.join(os.path.dirname(__file__), "ef_sweep.log")
        print(f"No log path given; defaulting to {log_path}")
    else:
        log_path = sys.argv[1]

    if not os.path.exists(log_path):
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    data, caps = parse_log(log_path)
    if not data:
        print("No RESULT lines found in log. Did the sweep finish?", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {sum(len(v['ef']) for v in data.values())} data points "
          f"across {len(data)} dataset(s).")
    if caps:
        print("Natural caps: " + ", ".join(f"{k}={v}" for k, v in caps.items()))

    out_dir  = os.path.dirname(os.path.abspath(log_path))
    out_3d   = os.path.join(out_dir, "ef_sweep_3d.png")
    out_2d   = os.path.join(out_dir, "ef_sweep_2d.png")

    print_summary(data, caps)
    plot_3d(data, caps, out_3d)
    plot_2d(data, caps, out_2d)
