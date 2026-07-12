#!/usr/bin/env python3
"""visualize_gamma_sweep.py

Parse the structured [GAMMA_SWEEP] log lines emitted by gamma_sweep and
produce a multi-panel figure:

  Row 0:  avg_recall   vs gamma   (one subplot per dataset)
  Row 1:  latency (ms) vs gamma   (one subplot per dataset)
  Row 2:  WAE          vs gamma   (one subplot per dataset)

Usage:
    # Run the binary and pipe its stdout to a log file, then visualize:
    EXPERIMENTS_ROOT=/data ./build/gamma_sweep 2>&1 | tee gamma_sweep.log
    python3 visualize_gamma_sweep.py gamma_sweep.log

    # Or read from stdin:
    cat gamma_sweep.log | python3 visualize_gamma_sweep.py

Output: gamma_sweep_plot.png (same directory as this script)
"""

import sys
import re
import os
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data structure ─────────────────────────────────────────────────────────────
# rows keyed by (dataset, gamma) -> {avg_recall, p5_recall, p1_recall, median_time_ms, wae}
Row = collections.namedtuple("Row", ["gamma", "avg_recall", "p5_recall", "p1_recall", "median_time_ms", "wae"])

# ── Parser ─────────────────────────────────────────────────────────────────────
PATTERN = re.compile(
    r"\[GAMMA_SWEEP\]\s+"
    r"dataset=(?P<dataset>\S+)\s+"
    r"gamma=(?P<gamma>[0-9.eE+\-]+)\s+"
    r"avg_recall=(?P<avg_recall>[0-9.eE+\-]+)\s+"
    r"p5_recall=(?P<p5_recall>[0-9.eE+\-]+)\s+"
    r"p1_recall=(?P<p1_recall>[0-9.eE+\-]+)\s+"
    r"median_time_ms=(?P<median_time_ms>[0-9]+)\s+"
    r"wae=(?P<wae>[0-9]+)"
)


def parse(source) -> dict[str, list[Row]]:
    data: dict[str, list[Row]] = collections.defaultdict(list)
    for line in source:
        m = PATTERN.search(line)
        if m:
            data[m.group("dataset")].append(Row(
                gamma=float(m.group("gamma")),
                avg_recall=float(m.group("avg_recall")),
                p5_recall=float(m.group("p5_recall")),
                p1_recall=float(m.group("p1_recall")),
                median_time_ms=int(m.group("median_time_ms")),
                wae=int(m.group("wae")),
            ))
    # Sort each dataset by gamma
    for ds in data:
        data[ds].sort(key=lambda r: r.gamma)
    return dict(data)


# ── Plotting ───────────────────────────────────────────────────────────────────
COLORS = {
    "deep-image-96-angular": "#E63946",
    "glove-100-angular":     "#2A9D8F",
    "sift-128-euclidean":    "#457B9D",
}
DEFAULT_COLOR = "#666666"

DATASET_LABELS = {
    "deep-image-96-angular": "Deep-Image-96 (angular)",
    "glove-100-angular":     "GloVe-100 (angular)",
    "sift-128-euclidean":    "SIFT-128 (euclidean)",
}


def plot(data: dict[str, list[Row]], out_path: str) -> None:
    datasets = sorted(data.keys())
    n = len(datasets)
    if n == 0:
        print("No [GAMMA_SWEEP] lines found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    ROWS = 3
    fig, axes = plt.subplots(ROWS, n, figsize=(6 * n, 4 * ROWS), squeeze=False)
    fig.suptitle("Gamma (decay constant) sweep — AdaEF HNSW\n"
                 "All other parameters held constant (α=0.25, k=100, target recall=0.95)",
                 fontsize=14, y=1.01)

    row_titles = ["Avg Recall", "Median Search Time (ms)", "Weighted Avg EF (WAE)"]
    y_keys     = ["avg_recall", "median_time_ms", "wae"]

    default_gamma = 16.0

    for col, ds in enumerate(datasets):
        rows = data[ds]
        gammas   = [r.gamma           for r in rows]
        recalls  = [r.avg_recall      for r in rows]
        p5_recalls = [r.p5_recall     for r in rows]
        p1_recalls = [r.p1_recall     for r in rows]
        times    = [r.median_time_ms  for r in rows]
        waes     = [r.wae             for r in rows]
        y_series = [recalls, times, waes]

        color = COLORS.get(ds, DEFAULT_COLOR)
        label = DATASET_LABELS.get(ds, ds)

        for row_idx, (ax, ys, rtitle) in enumerate(zip(axes[:, col], y_series, row_titles)):
            if row_idx == 0:
                # Plot all three recall metrics
                ax.plot(gammas, ys, marker="o", linewidth=2, color=color, label=f"Avg Recall")
                ax.plot(gammas, p5_recalls, marker="s", linewidth=1.5, linestyle="--", color=color, alpha=0.8, label="5th %ile")
                ax.plot(gammas, p1_recalls, marker="^", linewidth=1.5, linestyle=":", color=color, alpha=0.6, label="1st %ile")
            else:
                ax.plot(gammas, ys, marker="o", linewidth=2, color=color, label=label)

            # Highlight the default gamma=16
            if default_gamma in gammas:
                idx = gammas.index(default_gamma)
                ax.axvline(x=default_gamma, color="gray", linestyle="--", linewidth=1,
                           alpha=0.7, label=f"default γ={default_gamma:.0f}")
                if row_idx == 0:
                    ax.scatter([default_gamma]*3, [ys[idx], p5_recalls[idx], p1_recalls[idx]],
                               s=100, zorder=5, color=color,
                               edgecolors="black", linewidths=1.2)
                else:
                    ax.scatter([default_gamma], [ys[idx]],
                               s=120, zorder=5, color=color,
                               edgecolors="black", linewidths=1.5)

            # Axes labels
            ax.set_xlabel("γ (gamma)", fontsize=11)
            ax.set_ylabel(rtitle, fontsize=11)
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{x:g}"
            ))
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

            # Only set title at row 0
            if row_idx == 0:
                ax.set_title(label, fontsize=12, fontweight="bold")

            # Target recall line in row 0
            if row_idx == 0:
                ax.axhline(y=0.95, color="orange", linestyle="-.", linewidth=1.5,
                           label="target recall=0.95")
                min_y = min(min(p1_recalls), min(p5_recalls), min(recalls))
                max_y = max(max(p1_recalls), max(p5_recalls), max(recalls))
                ax.set_ylim(max(0, min_y - 0.02), min(1.01, max_y + 0.02))

            ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")


def print_summary(data: dict[str, list[Row]]) -> None:
    print("\n─── Summary ───────────────────────────────────────────────────────")
    print(f"{'Dataset':<25} {'γ':>6} {'Avg Recall':>10} {'p5 Recall':>10} {'p1 Recall':>10} {'Time(ms)':>9} {'WAE':>5}")
    print("─" * 82)
    for ds in sorted(data.keys()):
        for r in data[ds]:
            marker = " ←" if r.gamma == 16.0 else ""
            print(f"{ds:<25} {r.gamma:>6.1f} {r.avg_recall:>10.4f} {r.p5_recall:>10.4f} {r.p1_recall:>10.4f} "
                  f"{r.median_time_ms:>9} {r.wae:>5}{marker}")
    print("─" * 82)


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        with open(log_path, "r") as f:
            data = parse(f)
    else:
        data = parse(sys.stdin)

    if not data:
        print("[visualize_gamma_sweep] No [GAMMA_SWEEP] lines found in input.", file=sys.stderr)
        print("  Make sure to run: EXPERIMENTS_ROOT=... ./build/gamma_sweep 2>&1 | tee gamma_sweep.log",
              file=sys.stderr)
        sys.exit(1)

    print_summary(data)

    out_dir  = Path(__file__).parent
    out_path = str(out_dir / "gamma_sweep_plot.png")
    plot(data, out_path)


if __name__ == "__main__":
    main()
