import math
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

BASE_LOG = "research/log/output_shiro_k100.log"

ABLATIONS = {
    "hops": {
        "log": "research/log/output_shiro_k100.log",
        "title": "Visited List Size (1-hop / 2-hop / 3-hop)",
        "marker_re": r"^Visited list size:\s+(\d+)",
        "data_re": r"^\d+,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),",
        "label_fn": lambda v: {33: "1-hop", 1025: "2-hop", 32769: "3-hop"}.get(
            int(v), v
        ),
        "x_label": "Visited list size",
        "sort_num": True,
    },
    "gamma": {
        "log": "research/log/output_shiro_k100.log",
        "title": "Weighted Decay Function γ Ablation",
        "marker_re": r"^---\s*Gamma:\s+([\d.]+)\s*---",
        "data_re": r"^\d+,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),",
        "label_fn": str,
        "x_label": "Gamma",
        "sort_num": True,
    },
    "alpha": {
        "log": "research/log/output_shiro_alpha.log",
        "title": "Truncation Ratio α Ablation",
        "marker_re": r"^---\s*Alpha:\s+([\d.]+)\s*---",
        "data_re": r"^\d+,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),",
        "label_fn": str,
        "x_label": "Alpha",
        "sort_num": True,
    },
    "n_convergence_buckets": {
        "log": "research/log/output_shiro_k100.log",
        "title": "n_convergence_buckets Ablation",
        "marker_re": r"^---\s*n_convergence_buckets:\s+(\d+)\s*---",
        "data_re": r"^\d+,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),",
        "label_fn": str,
        "x_label": "n_convergence_buckets",
        "sort_num": True,
    },
    "min_q": {
        "log": "research/log/output_shiro_k100.log",
        "title": "min_queries_per_score Ablation",
        "marker_re": r"^---\s*min_queries_per_score:\s+(\d+)\s*---",
        "data_re": r"^\d+,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),",
        "label_fn": str,
        "x_label": "min_queries_per_score",
        "sort_num": True,
    },
    "sampling_size": {
        "log": "research/log/output_shiro_k100.log",
        "title": "Sampling Size Ablation",
        "marker_re": r"^Sampling size:\s+(\d+)",
        "data_re": r"^\d+,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),",
        "label_fn": str,
        "x_label": "Sampling size",
        "sort_num": True,
    },
}

METRIC_STYLE = {
    "avg": {"marker": "o", "size": 160, "label": "Avg"},
    "p05": {"marker": "s", "size": 100, "label": "P05"},
    "p01": {"marker": "^", "size": 100, "label": "P01"},
}


def parse_ablation(cfg):
    records = {}
    current_ds = None
    current_val = None
    marker_re = re.compile(cfg["marker_re"])
    data_re = re.compile(cfg["data_re"])
    with open(cfg["log"]) as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"^Dataset:\s+(.+)$", line)
            if m:
                current_ds = m.group(1).strip()
                records.setdefault(current_ds, {})
                current_val = None
                continue
            m = marker_re.match(line)
            if m:
                current_val = cfg["label_fn"](m.group(1).strip())
                continue
            m = data_re.match(line)
            if m and current_ds and current_val is not None:
                if current_val not in records[current_ds]:
                    records[current_ds][current_val] = {
                        "time_s": float(m.group(1)) / 1000.0,
                        "avg": float(m.group(2)),
                        "p05": float(m.group(3)),
                        "p01": float(m.group(4)),
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


def plot_ablation(name, cfg, base_data, out_dir):
    records = parse_ablation(cfg)
    if not records or not any(records.values()):
        print(f"  No data parsed for {name}, skipping.")
        return

    datasets = sorted(records.keys())
    n = len(datasets)
    ncols = 3
    nrows = math.ceil(n / ncols)

    all_param_vals = sorted(
        {k for ds_rows in records.values() for k in ds_rows},
        key=lambda v: (
            (0, float(v))
            if cfg.get("sort_num") and re.match(r"^[+-]?[\d.]+$", str(v))
            else (1, str(v))
        ),
    )
    cmap = matplotlib.colormaps["tab10"]
    color_map = {
        v: cmap(i / max(len(all_param_vals) - 1, 1))
        for i, v in enumerate(all_param_vals)
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for idx, ds in enumerate(datasets):
        ax = axes_flat[idx]
        ds_rows = records[ds]

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
            ax.fill_between(
                xs_bl, p01s_bl, avgs_bl, color="silver", alpha=0.25, zorder=1
            )

        hop_times = [row["time_s"] for row in ds_rows.values()]
        all_times = ([r["time_s"] for r in bl] if bl else []) + hop_times
        if all_times:
            span = max(all_times) - min(all_times)
            margin = max(span * 0.08, 0.5)
            ax.set_xlim(max(0, min(all_times) - margin), max(all_times) + margin)

        for pval in all_param_vals:
            row = ds_rows.get(pval)
            if row is None:
                continue
            c = color_map[pval]
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
                str(pval),
                xy=(t, row["avg"]),
                xytext=(0, 10),
                textcoords="offset points",
                va="center",
                fontsize=8,
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
            color=color_map[v],
            lw=3,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            label=f"{cfg['x_label']}={v}",
        )
        for v in all_param_vals
    ] + [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            markersize=7,
            markeredgecolor="white",
            ls="none",
            label="Avg",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="s",
            markersize=6,
            markeredgecolor="white",
            ls="none",
            label="P05",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="^",
            markersize=6,
            markeredgecolor="white",
            ls="none",
            label="P01",
        ),
        Line2D([0], [0], color="silver", lw=2, label="baseline ef sweep"),
        Line2D([0], [0], color="#e31a1c", ls="-.", lw=1.5, label="target 0.95"),
    ]
    ncol_leg = min(len(legend_handles), 8)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=ncol_leg,
        fontsize=8,
        framealpha=0.95,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        f"{cfg['title']}\n○=Avg  □=P05  △=P01  |  Bar=Avg–P01 spread  |  Grey band=baseline ef sweep",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0.07, 1, 1))

    out_path = os.path.join(out_dir, f"ablation_{name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {out_path}")


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(ABLATIONS.keys())
    out_dir = "research/img/ablation"
    os.makedirs(out_dir, exist_ok=True)
    base_data = parse_baseline(BASE_LOG) if os.path.exists(BASE_LOG) else {}

    for name in requested:
        if name not in ABLATIONS:
            print(f"Unknown ablation '{name}'. Available: {list(ABLATIONS.keys())}")
            continue
        cfg = ABLATIONS[name]
        if not os.path.exists(cfg["log"]):
            print(f"  Log not found for '{name}': {cfg['log']}, skipping.")
            continue
        print(f"Plotting ablation: {name}")
        plot_ablation(name, cfg, base_data, out_dir)


main()
