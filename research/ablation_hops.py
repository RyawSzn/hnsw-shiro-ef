import math
import os
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

HOP_LOG = "research/log/ablation_visited_list_size.log"
BASE_LOG = "research/log/output_shiro_new0.log"

HOP_ORDER = [513, 1025, 32769]
HOP_LABELS = {513: "1.5-hop", 1025: "2-hop", 32769: "3-hop"}
HOP_COLORS = {"1.5-hop": "#2166ac", "2-hop": "#d6604d", "3-hop": "#1a9641"}

METRIC_STYLE = {
    "avg": {"marker": "o", "size": 160, "label": "Avg"},
    "p05": {"marker": "s", "size": 100, "label": "P05"},
    "p01": {"marker": "^", "size": 100, "label": "P01"},
}


def parse_hops(path):
    records = {}
    current_ds, current_size = None, None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"^Dataset:\s+(.+)$", line)
            if m:
                current_ds = m.group(1).strip()
                records.setdefault(current_ds, {})
                continue
            m = re.match(r"^Visited list size:\s+(\d+)", line)
            if m:
                current_size = int(m.group(1))
                continue
            m = re.match(
                r"^(\d+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),", line
            )
            if m and current_ds and current_size and int(m.group(1)) == current_size:
                records[current_ds][current_size] = {
                    "time_s": float(m.group(2)) / 1000.0,
                    "avg": float(m.group(3)),
                    "p05": float(m.group(4)),
                    "p01": float(m.group(5)),
                }
    return records


def parse_baseline(path):
    baseline = {}
    current_ds = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"^Dataset:\s+(.+)$", line)
            if m:
                current_ds = m.group(1).strip()
                baseline.setdefault(current_ds, [])
                continue
            m = re.match(r"^\d+,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),", line)
            if m and current_ds:
                baseline[current_ds].append(
                    {
                        "time_s": float(m.group(1)) / 1000.0,
                        "avg": float(m.group(2)),
                        "p05": float(m.group(3)),
                        "p01": float(m.group(4)),
                    }
                )
    return baseline


hop_data = parse_hops(HOP_LOG)
base_data = parse_baseline(BASE_LOG)

datasets = sorted(hop_data.keys())
n = len(datasets)
ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
axes_flat = axes.flatten() if n > 1 else [axes]
for i in range(n, len(axes_flat)):
    axes_flat[i].set_visible(False)

for idx, ds in enumerate(datasets):
    ax = axes_flat[idx]
    hop_rows = hop_data.get(ds, {})

    bl = sorted(base_data.get(ds, []), key=lambda r: r["time_s"])
    if bl:
        seen_max = -1.0
        bl_mono = []
        for r in bl:
            if r["avg"] >= seen_max:
                bl_mono.append(r)
                seen_max = r["avg"]
        bl = bl_mono
    if bl:
        xs_bl = [r["time_s"] for r in bl]
        avgs_bl = [r["avg"] for r in bl]
        p01s_bl = [r["p01"] for r in bl]
        ax.plot(xs_bl, avgs_bl, color="silver", lw=1.6, zorder=1)
        ax.fill_between(xs_bl, p01s_bl, avgs_bl, color="silver", alpha=0.25, zorder=1)

    hop_times = [row["time_s"] for row in hop_rows.values()]
    all_times = ([r["time_s"] for r in bl] if bl else []) + hop_times
    if all_times:
        span = max(all_times) - min(all_times)
        margin = max(span * 0.08, 0.5)
        ax.set_xlim(max(0, min(all_times) - margin), max(all_times) + margin)

    for size in HOP_ORDER:
        row = hop_rows.get(size)
        if row is None:
            continue
        hop = HOP_LABELS[size]
        c = HOP_COLORS[hop]
        t = row["time_s"]

        ax.plot(
            [t, t],
            [row["p01"], row["avg"]],
            color=c,
            lw=3.0,
            solid_capstyle="round",
            zorder=4,
        )

        for mkey, ms in METRIC_STYLE.items():
            ax.scatter(
                [t],
                [row[mkey]],
                color=c,
                marker=ms["marker"],
                s=ms["size"],
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            )

        ax.annotate(
            hop,
            xy=(t, row["avg"]),
            xytext=(0, 10),
            textcoords="offset points",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=c,
        )

    ax.axhline(y=0.95, color="#e31a1c", ls="-.", lw=1.2, alpha=0.8, zorder=6)

    ax.set_ylim(0.70, 1.01)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    ax.set_title(ds, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Latency (s)", fontsize=10)
    ax.set_ylabel("Recall@100", fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

legend_handles = [
    Line2D(
        [0],
        [0],
        color=HOP_COLORS[h],
        lw=3,
        marker="o",
        markersize=8,
        markeredgecolor="white",
        label=h,
    )
    for h in HOP_COLORS
] + [
    Line2D(
        [0],
        [0],
        color="black",
        marker="o",
        markersize=8,
        markeredgecolor="white",
        ls="none",
        label="Avg",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        marker="s",
        markersize=7,
        markeredgecolor="white",
        ls="none",
        label="P05",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        marker="^",
        markersize=7,
        markeredgecolor="white",
        ls="none",
        label="P01",
    ),
    Line2D([0], [0], color="silver", lw=2, label="baseline ef sweep"),
    Line2D([0], [0], color="#e31a1c", ls="-.", lw=1.5, label="target 0.95"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(legend_handles),
    fontsize=9,
    framealpha=0.95,
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, 0.0),
)

fig.suptitle(
    "Visited List Size Ablation: 1.5-hop / 2-hop / 3-hop\n"
    "○=Avg  □=P05  △=P01  |  Bar=Avg–P01 spread  |  Grey band=baseline ef sweep",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout(rect=(0, 0.06, 1, 1))

os.makedirs("research/img", exist_ok=True)
plt.savefig("research/img/ablation_hops.png", dpi=150, bbox_inches="tight")
print("Saved to research/img/ablation_hops.png")
