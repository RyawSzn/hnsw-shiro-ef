#!/usr/bin/env python3
"""
affine_calibrate_ada.py
========================

Direction of this calibration
------------------------------
Earlier scripts scaled SHIRO's numbers down onto ada's hardware scale
(shiro_calibrated = shiro_raw / k). This script does the opposite: it keeps
shiro as the reference machine and ADJUSTS ADA's numbers up onto shiro's
hardware scale, using a full affine model instead of a pure through-origin
scale.

Model
-----
For each dataset shared by both logs, fit (via ordinary least squares) an
affine map from ada's baseline latency to shiro's baseline latency:

    shiro_ns  ~=  a_dataset + b_dataset * ada_ns

This is fit directly from the paired (ada_ns, shiro_ns) points at every `ef`
value both logs tested for that dataset (recall is identical at each ef, so
these pairs are a valid apples-to-apples speed comparison -- see
calibrate.py for the parity check).

Because the fit already predicts "what shiro's latency would be for a given
ada latency", applying it directly to ada's OWN latency numbers is exactly
the adjustment we want:

    ada_adjusted_ns = a_dataset + b_dataset * ada_ns

This is done for:
  - every fixed-ef baseline point ("Testing ef: N" -> "=== Summary for ef N ===")
  - every per-iteration adaptive-method summary ("=== Summary (Iteration i/3) ===")
  - the "=== Final Summary (Median Iteration) ===" block

For the 2 (dataset) x 2 (Average Latency / Total Latency) numbers in each
such block we rewrite the ORIGINAL LOG TEXT in place, so the output is a
complete, valid log file ("affined log") -- not just a CSV of numbers -- that
reads exactly like ada-ef's log would have, had it been produced on shiro's
machine.

Datasets ada never shares with shiro (none in this data, but handled
defensively) fall back to the POOLED affine fit across all shared datasets,
and are flagged in the console output as lower-confidence.

Outputs
-------
  - output_ada_full_AFFINED.log   the rewritten ada log, adjusted to shiro's HW
  - affine_fit_parameters.csv     per-dataset (a, b, R^2) fit parameters
  - affine_method_comparison.csv  adjusted ada vs raw shiro adaptive-method latency
  - affine_calibration_plots.png  visual sanity check (before/after, fit params,
                                   adaptive-method comparison)
"""

import csv
import re
import statistics

from parse_logs import parse_log

ADA_LOG = "research/log/output_ada_full.log"
SHIRO_LOG = "research/log/output_shiro_full.log"
OUT_LOG = "research/log/output_ada_full_AFFINED.log"


# ----------------------------------------------------------------------
# 1. Fit per-dataset affine models from the shared baseline sweep
# ----------------------------------------------------------------------


def fit_affine(xs, ys):
    """OLS fit y = a + b*x. Returns (a, b, r2)."""
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    pred = [a + b * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a, b, r2


def build_dataset_fits():
    b_ada, _ = parse_log(ADA_LOG, "ada")
    b_shiro, _ = parse_log(SHIRO_LOG, "shiro")
    idx_ada = {(r["dataset"], r["ef"]): r for r in b_ada}
    idx_shiro = {(r["dataset"], r["ef"]): r for r in b_shiro}
    common = sorted(set(idx_ada) & set(idx_shiro))

    by_ds = {}
    for k in common:
        by_ds.setdefault(k[0], []).append(k)

    fits = {}
    all_x, all_y = [], []
    rows = []
    print(
        f"{'dataset':28s} {'n_ef':>5s} {'a (intercept, ns)':>18s} {'b (slope)':>10s} {'R2':>7s}"
    )
    for ds, keys in sorted(by_ds.items()):
        xs = [idx_ada[k]["latency_ns"] for k in keys]
        ys = [idx_shiro[k]["latency_ns"] for k in keys]
        a, b, r2 = fit_affine(xs, ys)
        fits[ds] = (a, b, r2)
        all_x.extend(xs)
        all_y.extend(ys)
        print(f"{ds:28s} {len(keys):5d} {a:18.1f} {b:10.4f} {r2:7.4f}")
        rows.append(
            {
                "dataset": ds,
                "n_matched_ef": len(keys),
                "a_intercept_ns": a,
                "b_slope": b,
                "r2": r2,
            }
        )

    a_pool, b_pool, r2_pool = fit_affine(all_x, all_y)
    print(
        f"{'POOLED':28s} {len(all_x):5d} {a_pool:18.1f} {b_pool:10.4f} {r2_pool:7.4f}"
    )
    rows.append(
        {
            "dataset": "POOLED",
            "n_matched_ef": len(all_x),
            "a_intercept_ns": a_pool,
            "b_slope": b_pool,
            "r2": r2_pool,
        }
    )

    with open("research/csv/affine_fit_parameters.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["dataset", "n_matched_ef", "a_intercept_ns", "b_slope", "r2"]
        )
        w.writeheader()
        w.writerows(rows)
    print("Wrote: research/csv/affine_fit_parameters.csv\n")

    return fits, (a_pool, b_pool, r2_pool)


# ----------------------------------------------------------------------
# 2. Rewrite ada's own log text, adjusting every relevant latency number
#    through that dataset's affine map
# ----------------------------------------------------------------------


def fmt(value):
    """Match the log's own number style: plain below 1e6, '%g' scientific above."""
    return format(float(value), "g")


def adjust_latency_ns(value_ns, a, b):
    return a + b * value_ns


def rewrite_log(fits, pooled_fit):
    with open(ADA_LOG, "r", errors="ignore") as f:
        lines = f.readlines()

    dataset_re = re.compile(r"^Dataset:\s*(\S+)")
    avg_lat_re = re.compile(r"(Average Latency:\s*)([0-9.eE+\-]+)(\s*ns)")
    total_lat_re = re.compile(r"(Total Latency:\s*)([0-9.eE+\-]+)(\s*seconds)")

    current_dataset = None
    replacements = {}  # line_index -> new full line text
    n_adjusted_pairs = 0

    # Pass 1: find every 'Average Latency' line, its paired 'Total Latency'
    # line (which may sit either just before or just after it), and compute
    # both replacements up front -- so write order in the source file can
    # never cause a stale/unpatched line to slip through.
    for i, line in enumerate(lines):
        m = dataset_re.match(line)
        if m:
            current_dataset = m.group(1)
            continue

        m = avg_lat_re.search(line)
        if not m or current_dataset is None:
            continue

        if current_dataset in fits:
            a, b, r2 = fits[current_dataset]
        else:
            a, b, r2 = pooled_fit

        orig_avg_ns = float(m.group(2))
        new_avg_ns = adjust_latency_ns(orig_avg_ns, a, b)
        replacements[i] = avg_lat_re.sub(
            lambda mm: mm.group(1) + fmt(new_avg_ns) + mm.group(3), line
        )

        total_idx = None
        for j in (i - 1, i + 1, i - 2, i + 2):
            if 0 <= j < len(lines) and total_lat_re.search(lines[j]):
                total_idx = j
                break

        if total_idx is not None:
            tm = total_lat_re.search(lines[total_idx])
            orig_total_s = float(tm.group(2))
            n_queries = round(orig_total_s * 1e9 / orig_avg_ns) if orig_avg_ns else None
            if n_queries:
                new_total_s = new_avg_ns * n_queries / 1e9
                replacements[total_idx] = total_lat_re.sub(
                    lambda mm: mm.group(1) + fmt(new_total_s) + mm.group(3),
                    lines[total_idx],
                )

        n_adjusted_pairs += 1

    # Pass 2: rebuild the file, applying every precomputed replacement.
    out_lines = [replacements.get(i, line) for i, line in enumerate(lines)]

    with open(OUT_LOG, "w") as f:
        f.writelines(out_lines)

    print(
        f"Adjusted {n_adjusted_pairs} 'Average Latency' lines and "
        f"{len(replacements) - n_adjusted_pairs} paired 'Total Latency' lines."
    )
    print(f"Wrote: {OUT_LOG}")


# ----------------------------------------------------------------------
# 3. Adjusted vs raw comparison for the adaptive-method result specifically
# ----------------------------------------------------------------------


def compare_methods(fits, pooled_fit):
    _, m_ada = parse_log(ADA_LOG, "ada")
    _, m_shiro = parse_log(SHIRO_LOG, "shiro")
    m_shiro_d = {r["dataset"]: r for r in m_shiro}

    rows = []
    print(
        f"\n{'dataset':28s} {'ada_raw_ns':>12s} {'ada_adjusted_ns':>16s} "
        f"{'shiro_raw_ns':>13s} {'ada_recall':>11s} {'shiro_recall':>13s}"
    )
    for r in m_ada:
        ds = r["dataset"]
        if ds in fits:
            a, b, r2 = fits[ds]
            confidence = "dataset-specific"
        else:
            a, b, r2 = pooled_fit
            confidence = "pooled-fallback"
        adjusted_ns = adjust_latency_ns(r["avg_latency_ns"], a, b)
        s = m_shiro_d.get(ds)
        row = {
            "dataset": ds,
            "ada_recall": r["avg_recall"],
            "ada_latency_ns_raw": r["avg_latency_ns"],
            "ada_latency_ns_adjusted_to_shiro_hw": adjusted_ns,
            "shiro_recall": s["avg_recall"] if s else None,
            "shiro_latency_ns_raw": s["avg_latency_ns"] if s else None,
            "affine_a": a,
            "affine_b": b,
            "affine_r2": r2,
            "confidence": confidence,
        }
        rows.append(row)
        print(
            f"{ds:28s} {r['avg_latency_ns']:12.0f} {adjusted_ns:16.0f} "
            f"{(s['avg_latency_ns'] if s else float('nan')):13.0f} "
            f"{r['avg_recall']:11.4f} "
            f"{(s['avg_recall'] if s else float('nan')):13.4f}"
        )

    with open("research/csv/affine_method_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote: research/csv/affine_method_comparison.csv")


def main():
    fits, pooled_fit = build_dataset_fits()
    rewrite_log(fits, pooled_fit)
    compare_methods(fits, pooled_fit)
    make_plots(fits, pooled_fit)


# ----------------------------------------------------------------------
# 4. Visual sanity check
# ----------------------------------------------------------------------


def make_plots(fits, pooled_fit):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b_ada, m_ada = parse_log(ADA_LOG, "ada")
    b_shiro, m_shiro = parse_log(SHIRO_LOG, "shiro")
    idx_ada = {(r["dataset"], r["ef"]): r for r in b_ada}
    idx_shiro = {(r["dataset"], r["ef"]): r for r in b_shiro}
    common = sorted(set(idx_ada) & set(idx_shiro))
    by_ds = {}
    for k in common:
        by_ds.setdefault(k[0], []).append(k)
    m_ada_d = {r["dataset"]: r for r in m_ada}
    m_shiro_d = {r["dataset"]: r for r in m_shiro}

    mosaic = """
    AABB
    CDEE
    """
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(16, 10))

    # (A) Raw baseline latency curves: ada vs shiro, per dataset -- shows the mismatch
    ax = axd["A"]
    for dataset, keys in sorted(by_ds.items()):
        keys_sorted = sorted(keys, key=lambda k: k[1])
        efs = [k[1] for k in keys_sorted]
        ada_lat = [idx_ada[k]["latency_ns"] / 1e6 for k in keys_sorted]
        shiro_lat = [idx_shiro[k]["latency_ns"] / 1e6 for k in keys_sorted]
        (line,) = ax.plot(efs, shiro_lat, "o-", label=f"{dataset} (shiro)", alpha=0.8)
        ax.plot(
            efs,
            ada_lat,
            "x--",
            color=line.get_color(),
            alpha=0.8,
            label=f"{dataset} (ada raw)",
        )
    ax.set_xlabel("ef")
    ax.set_ylabel("latency (ms)")
    ax.set_title("BEFORE calibration: baseline latency, shiro vs ada (raw)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    # (B) Calibrated baseline latency curves -- ada lifted onto shiro's scale
    ax = axd["B"]
    for dataset, keys in sorted(by_ds.items()):
        keys_sorted = sorted(keys, key=lambda k: k[1])
        efs = [k[1] for k in keys_sorted]
        shiro_lat = [idx_shiro[k]["latency_ns"] / 1e6 for k in keys_sorted]
        a, b, _ = fits[dataset]
        ada_adj = [
            adjust_latency_ns(idx_ada[k]["latency_ns"], a, b) / 1e6 for k in keys_sorted
        ]
        (line,) = ax.plot(efs, shiro_lat, "o-", label=f"{dataset} (shiro)", alpha=0.8)
        ax.plot(
            efs,
            ada_adj,
            "x--",
            color=line.get_color(),
            alpha=0.8,
            label=f"{dataset} (ada adjusted)",
        )
    ax.set_xlabel("ef")
    ax.set_ylabel("latency (ms)")
    ax.set_title("AFTER calibration: baseline latency, shiro vs ada (adjusted)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    datasets = sorted(fits.keys())

    # (C) Slope per dataset, own panel, own scale
    ax = axd["C"]
    b_vals = [fits[d][1] for d in datasets]
    bars = ax.bar(datasets, b_vals, color="steelblue")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    for rect, v in zip(bars, b_vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + 0.01,
            f"{v:.3f}",
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("slope b  (shiro_ns / ada_ns)")
    ax.set_title("Fitted slope per dataset")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")

    # (D) Intercept per dataset, own panel, own scale
    ax = axd["D"]
    a_vals_ms = [fits[d][0] / 1e6 for d in datasets]
    bars = ax.bar(datasets, a_vals_ms, color="indianred")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    for rect, v in zip(bars, a_vals_ms):
        offset = 0.01 * (max(a_vals_ms) - min(a_vals_ms) + 1e-9)
        va = "bottom" if v >= 0 else "top"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + (offset if v >= 0 else -offset),
            f"{v:+.3f}",
            ha="center",
            va=va,
            fontsize=8,
        )
    ax.set_ylabel("intercept a (ms)")
    ax.set_title("Fitted intercept per dataset")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")

    # (E) Adaptive-method comparison: ada raw vs ada adjusted vs shiro raw
    ax = axd["E"]
    ds_order = [d for d in sorted(set(m_ada_d) | set(m_shiro_d))]
    ada_raw, ada_adj, shiro_raw = [], [], []
    for d in ds_order:
        a_row = m_ada_d.get(d)
        s_row = m_shiro_d.get(d)
        ada_raw.append(a_row["avg_latency_ns"] / 1e6 if a_row else 0)
        if a_row:
            a, b, _ = fits.get(d, pooled_fit)
            ada_adj.append(adjust_latency_ns(a_row["avg_latency_ns"], a, b) / 1e6)
        else:
            ada_adj.append(0)
        shiro_raw.append(s_row["avg_latency_ns"] / 1e6 if s_row else 0)

    x = range(len(ds_order))
    width = 0.27
    ax.bar([i - width for i in x], ada_raw, width, label="ada (as-reported, raw)")
    ax.bar([i for i in x], ada_adj, width, label="ada (adjusted to shiro HW)")
    ax.bar([i + width for i in x], shiro_raw, width, label="shiro (as-reported)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ds_order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("adaptive-method avg latency (ms)")
    ax.set_title("Adaptive-method latency: ada raw vs ada adjusted vs shiro")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Affine calibration: adjusting ada onto shiro's hardware scale "
        "(shiro_ns = a + b\u00b7ada_ns)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("research/img/affine/affine_calibration_plots.png", dpi=150)
    print("Wrote: research/img/affine/affine_calibration_plots.png")


if __name__ == "__main__":
    main()
