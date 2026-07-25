import math
import os
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

LOG_METHOD_0 = "research/log/output_shiro_glove0.log"
LOG_METHOD_1 = "research/log/output_shiro_glove1.log"


def parse_shiro_log(filepath):
    baseline = []
    our_method = {}

    current_block = None
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()

                m_st = re.match(r"statisc_length, time, avg_recall", line)
                if m_st:
                    current_block = "our_method"
                    continue

                m_ef = re.match(r"ef, time, avg_recall", line)
                if m_ef:
                    current_block = "baseline"
                    continue

                m_data = re.match(
                    r"^(\d+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),", line
                )
                if m_data:
                    param = int(m_data.group(1))
                    time_s = float(m_data.group(2)) / 1000.0
                    avg = float(m_data.group(3))
                    p05 = float(m_data.group(4))
                    p01 = float(m_data.group(5))

                    if current_block == "our_method":
                        our_method = {
                            "time_s": time_s,
                            "avg": avg,
                            "p05": p05,
                            "p01": p01,
                        }
                    elif current_block == "baseline":
                        baseline.append(
                            {"time_s": time_s, "avg": avg, "p05": p05, "p01": p01}
                        )
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return our_method, baseline


method0_om, _ = parse_shiro_log(LOG_METHOD_0)
method1_om, baseline = parse_shiro_log(LOG_METHOD_1)

# Enforce monotonicity on baseline
baseline = sorted(baseline, key=lambda x: x["time_s"])
mono_baseline = []
seen_max_avg = -1.0
seen_max_p05 = -1.0
seen_max_p01 = -1.0
for r in baseline:
    if r["avg"] >= seen_max_avg:  # Use avg as the main monotonicity driver
        r["avg"] = max(r["avg"], seen_max_avg)
        r["p05"] = max(r["p05"], seen_max_p05)
        r["p01"] = max(r["p01"], seen_max_p01)
        mono_baseline.append(r)
        seen_max_avg = r["avg"]
        seen_max_p05 = r["p05"]
        seen_max_p01 = r["p01"]

bl_times = [r["time_s"] for r in mono_baseline]
bl_avg = [r["avg"] for r in mono_baseline]
bl_p05 = [r["p05"] for r in mono_baseline]
bl_p01 = [r["p01"] for r in mono_baseline]

fig, ax = plt.subplots(figsize=(8, 7))

# Plot baseline lines
ax.plot(
    bl_times,
    bl_avg,
    "-",
    color="tab:blue",
    label="Baseline - Avg",
    linewidth=2,
    alpha=0.3,
)
ax.plot(
    bl_times,
    bl_p05,
    "-",
    color="tab:green",
    label=r"Baseline - $\pi_{0.05}$",
    linewidth=2,
    alpha=0.3,
)
ax.plot(
    bl_times,
    bl_p01,
    "-",
    color="tab:orange",
    label=r"Baseline - $\pi_{0.01}$",
    linewidth=2,
    alpha=0.3,
)
ax.fill_between(bl_times, bl_p01, bl_avg, color="silver", alpha=0.15, zorder=1)

METRIC_STYLE = {
    "avg": {"marker": "o", "size": 160, "label": "Avg"},
    "p05": {"marker": "s", "size": 100, "label": "P05"},
    "p01": {"marker": "^", "size": 100, "label": "P01"},
}

METHODS = [
    {
        "name": "Method 0 (Uniform)",
        "data": method0_om,
        "color": "#d6604d",
        "offset": -1.0,
    },
    {
        "name": "Method 1 (Hard-first)",
        "data": method1_om,
        "color": "#2166ac",
        "offset": 1.0,
    },
]

for m in METHODS:
    d = m["data"]
    if not d:
        continue
    t = d["time_s"]
    c = m["color"]

    # Draw vertical spread bar
    ax.plot(
        [t, t], [d["p01"], d["avg"]], color=c, lw=3.0, solid_capstyle="round", zorder=4
    )

    # Draw metric markers
    for mkey, ms in METRIC_STYLE.items():
        ax.scatter(
            [t],
            [d[mkey]],
            color=c,
            marker=ms["marker"],
            s=ms["size"],
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )

    ax.annotate(
        m["name"] + f"\n({t:.1f}s)",
        xy=(t, d["avg"]),
        xytext=(-6 if m["offset"] < 0 else 6, 12),
        textcoords="offset points",
        ha="right" if m["offset"] < 0 else "left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=c,
    )

# Draw an arrow pointing out the P01 improvement
if method0_om and method1_om:
    ax.annotate(
        "Better performance\non hard queries",
        xy=(method1_om["time_s"], method1_om["p01"]),
        xytext=(method1_om["time_s"] + 8, method1_om["p01"] - 0.05),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )

ax.axhline(y=0.95, color="#e31a1c", ls="-.", lw=1.2, alpha=0.8, zorder=2)

ax.set_title(
    "Sampling Method Comparison on glove-100-angular",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel("Latency (s)", fontsize=11)
ax.set_ylabel("Recall@100", fontsize=11)

ax.set_ylim(0.70, 1.01)

all_times = bl_times.copy()
if method0_om:
    all_times.append(method0_om["time_s"])
if method1_om:
    all_times.append(method1_om["time_s"])

if all_times:
    ax.set_xlim(0, max(all_times) * 1.01)

ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.grid(True, ls="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    Line2D(
        [0],
        [0],
        color="#d6604d",
        lw=3,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        label="Method 0 (Uniform)",
    ),
    Line2D(
        [0],
        [0],
        color="#2166ac",
        lw=3,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        label="Method 1 (Hard-first)",
    ),
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
    Line2D([0], [0], color="silver", lw=2, label="Baseline (glove1)"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    fontsize=9,
    framealpha=0.95,
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, 0.0),
)

plt.tight_layout(rect=(0, 0.1, 1, 1))

out_dir = "research/img"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "compare_sampling.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
