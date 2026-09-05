#!/usr/bin/env python3
"""
Parse output_shiro_full.log and reproduce the "recall as price, time as volume"
indexed charts (line chart of recall indexed to a baseline = 100, bar chart of
average search time indexed the same way) for each parameter sweep found in
the log:

  - visit list size   (the un-marked sweep at the top of the log, keyed by
                        the `statisc_length` field itself: 33, 100, 250, ...)
  - Sampling size      (marked by lines like "Sampling size: 3000")
  - Gamma              (marked by lines like "--- Gamma: 16 ---")
  - Alpha              (marked by lines like "--- Alpha: 0.5 ---")
  - min_queries_per_score
  - n_convergence_buckets

Usage:
    python visualize_sweeps.py [path/to/output_shiro_full.log]

Requires: matplotlib (pip install matplotlib --break-system-packages)
Output:   PNG files written to ./sweep_plots/
"""

import os
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # safe for headless / no-display environments
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# CONFIG - tweak these without touching the parsing/plotting logic below
# --------------------------------------------------------------------------

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "research/log/output_shiro_full.log"
OUT_DIR = "research/img/sweep_plots"

# For each sweep: which parameter value counts as "current" (indexed to 100),
# and which values (if any) to drop before plotting. Use None for "keep all".
SWEEP_CONFIG = {
    "visit_list_size": {
        "baseline": 1025,
        "keep_only": [33, 1025, 32769],  # set to None to keep every value tested
    },
    "Sampling size": {
        "baseline": 3000,
        "keep_only": None,
    },
    "Gamma": {
        "baseline": 16,
        "keep_only": None,
    },
    "Alpha": {
        "baseline": 0.5,
        "keep_only": None,
    },
    "min_queries_per_score": {
        "baseline": 1,
        "keep_only": None,
    },
    "n_convergence_buckets": {
        "baseline": 10,
        "drop": [0],  # 0 buckets is a broken/degenerate config
        "keep_only": None,
    },
}

# marker regexes -> sweep name + value-parser
MARKER_PATTERNS = [
    ("Sampling size", re.compile(r"^Sampling size: (\d+)\s*$"), int),
    ("Gamma", re.compile(r"^--- Gamma: (\d+) ---\s*$"), int),
    ("Alpha", re.compile(r"^--- Alpha: ([\d.]+) ---\s*$"), float),
    (
        "min_queries_per_score",
        re.compile(r"^--- min_queries_per_score: (\d+) ---\s*$"),
        int,
    ),
    (
        "n_convergence_buckets",
        re.compile(r"^--- n_convergence_buckets: (\d+) ---\s*$"),
        int,
    ),
]

RESULT_LINE_RE = re.compile(r"^(?P<dataset>[\w.\-]+) experiment results:\s*$")

# --------------------------------------------------------------------------
# 1. Parse the log
# --------------------------------------------------------------------------


def parse_log(path):
    with open(path, "r") as f:
        lines = f.read().split("\n")

    # sweeps[sweep_name][dataset][param_value] = dict(avg, p5, p1, time_ms)
    sweeps = defaultdict(lambda: defaultdict(dict))

    pending = None  # (sweep_name, value) set by the most recent marker line
    i = 0
    while i < len(lines):
        line = lines[i]

        matched_marker = False
        for name, pattern, caster in MARKER_PATTERNS:
            m = pattern.match(line)
            if m:
                pending = (name, caster(m.group(1)))
                matched_marker = True
                break
        if matched_marker:
            i += 1
            continue

        m = RESULT_LINE_RE.match(line)
        if m:
            dataset = m.group("dataset")
            # data row is 2 lines below the "... experiment results:" line
            # (line+1 is typically a header/description line in this log format)
            data_line = lines[i + 2].strip()
            parts = [p.strip() for p in data_line.split(",")]
            statisc_length = int(parts[0])
            time_ms = int(parts[1])
            avg_recall = float(parts[2])
            p5 = float(parts[3])
            p1 = float(parts[4])

            row = {"time_ms": time_ms, "avg": avg_recall, "p5": p5, "p1": p1}

            if pending is not None:
                sweep_name, value = pending
                sweeps[sweep_name][dataset][value] = row
                pending = None  # each marker covers exactly one result block
            else:
                # unmarked sweep at the top of the log: keyed by statisc_length itself
                sweeps["visit_list_size"][dataset][statisc_length] = row

            i += 3
            continue

        i += 1

    return sweeps


# --------------------------------------------------------------------------
# 2. Aggregate across datasets (unweighted mean per parameter value)
# --------------------------------------------------------------------------


def aggregate(sweep_data):
    """sweep_data: dataset -> value -> row.  Returns value -> {avg, p5, p1, time_ms} averaged over datasets."""
    values = sorted({v for ds_rows in sweep_data.values() for v in ds_rows})
    agg = {}
    for v in values:
        avgs, p5s, p1s, times = [], [], [], []
        for ds, rows in sweep_data.items():
            if v not in rows:
                continue
            avgs.append(rows[v]["avg"])
            p5s.append(rows[v]["p5"])
            p1s.append(rows[v]["p1"])
            times.append(rows[v]["time_ms"])
        if not avgs:
            continue
        agg[v] = {
            "avg": statistics.mean(avgs),
            "p5": statistics.mean(p5s),
            "p1": statistics.mean(p1s),
            "time_ms": statistics.mean(times),
        }
    return agg


# --------------------------------------------------------------------------
# 3. Plot: line chart of recall indexed to baseline=100, bar chart of time
#    indexed the same way, stacked like a price/volume stock chart
# --------------------------------------------------------------------------


def plot_sweep(sweep_name, agg, baseline, keep_only=None, drop=None):
    values = sorted(agg.keys())
    if drop:
        values = [v for v in values if v not in drop]
    if keep_only:
        values = [v for v in values if v in keep_only]
    if baseline not in agg:
        print(
            f"  [!] baseline {baseline} not found in data for '{sweep_name}', skipping"
        )
        return
    if baseline not in values:
        values = sorted(values + [baseline])

    base = agg[baseline]
    labels = [str(v) for v in values]
    avg_idx = [agg[v]["avg"] / base["avg"] * 100 for v in values]
    p5_idx = [agg[v]["p5"] / base["p5"] * 100 for v in values]
    p1_idx = [agg[v]["p1"] / base["p1"] * 100 for v in values]
    time_idx = [agg[v]["time_ms"] / base["time_ms"] * 100 for v in values]

    current_i = values.index(baseline)

    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    x = range(len(values))
    ax_price.axhline(
        100,
        color="#c3c2b7",
        linestyle="--",
        linewidth=1,
        label=f"{baseline} baseline (=100)",
    )
    ax_price.plot(x, avg_idx, marker="o", color="#2a78d6", label="avg recall")
    ax_price.plot(
        x, p5_idx, marker="o", color="#eb6834", linestyle="--", label="p5 recall"
    )
    ax_price.plot(
        x, p1_idx, marker="o", color="#6250d6", linestyle=":", label="p1 recall"
    )
    ax_price.set_ylabel(f"recall index ({baseline} = 100)")
    ax_price.set_title(f"{sweep_name}: recall (indexed) and search time (indexed)")
    ax_price.legend(loc="best", fontsize=8)
    ax_price.grid(True, color="#e1e0d9")

    bar_colors = [
        "#BA7517" if i == current_i else "#B4B2A9" for i in range(len(values))
    ]
    ax_vol.bar(x, time_idx, color=bar_colors)
    ax_vol.axhline(100, color="#c3c2b7", linestyle="--", linewidth=1)
    time_min = min(time_idx)
    time_max = max(time_idx)
    margin = max((time_max - time_min) * 0.2, 1.0)
    ax_vol.set_ylim(bottom=time_min - margin)
    ax_vol.set_ylabel(f"time index ({baseline} = 100)")
    ax_vol.set_xticks(list(x))
    ax_vol.set_xticklabels(labels)
    ax_vol.grid(True, axis="y", color="#e1e0d9")

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{sweep_name.replace(' ', '_')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main():
    if not os.path.exists(LOG_PATH):
        print(f"Log file not found: {LOG_PATH}")
        print(
            "Pass the path as an argument: python visualize_sweeps.py /path/to/output_shiro_full.log"
        )
        sys.exit(1)

    print(f"Parsing {LOG_PATH} ...")
    sweeps = parse_log(LOG_PATH)

    for sweep_name, data in sweeps.items():
        cfg = SWEEP_CONFIG.get(sweep_name, {})
        baseline = cfg.get("baseline")
        if baseline is None:
            print(f"[skip] no baseline configured for '{sweep_name}' in SWEEP_CONFIG")
            continue
        print(
            f"[{sweep_name}] {len(data)} dataset(s), "
            f"{len({v for rows in data.values() for v in rows})} distinct value(s)"
        )
        agg = aggregate(data)
        plot_sweep(
            sweep_name,
            agg,
            baseline,
            keep_only=cfg.get("keep_only"),
            drop=cfg.get("drop"),
        )

    print(f"\nDone. PNGs are in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
