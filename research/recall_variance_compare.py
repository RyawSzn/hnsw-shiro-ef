#!/usr/bin/env python3
"""
recall_variance_compare.py
Compare recall variance: fixed-EF baseline vs Shiro adaptive-EF results.

Usage:
    python3 recall_variance_compare.py <dataset> <baseline_ef>

    dataset      : one of the dataset slugs (e.g. deep-image-96-angular)
    baseline_ef  : which EF row to pull from the baseline CSV (e.g. 100)

Examples:
    python3 recall_variance_compare.py deep-image-96-angular 100
    python3 recall_variance_compare.py glove-100-angular 200

Available datasets (auto-detected from csv_shiro/):
    Run with --list to print them.
"""

import sys
import csv
import math
import argparse
from pathlib import Path

CSV_DIR = Path(__file__).parent / "csv_shiro"


# ── helpers ──────────────────────────────────────────────────────────────────

def available_datasets() -> list[str]:
    slugs = []
    for p in sorted(CSV_DIR.glob("per_query_baseline_*.csv")):
        slug = p.stem.removeprefix("per_query_baseline_")
        if not slug.endswith("_recall_variance"):   # skip sidecar files
            slugs.append(slug)
    return slugs


def load_recalls_at_ef(path: Path, ef: int) -> list[float]:
    recalls: list[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["EF"]) == ef:
                recalls.append(float(row["Recall"]))
    return recalls


def load_all_recalls(path: Path) -> list[float]:
    """Load every Recall value regardless of EF (used for shiro results)."""
    with path.open(newline="") as f:
        return [float(row["Recall"]) for row in csv.DictReader(f)]


def available_efs(path: Path) -> list[int]:
    efs: set[int] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            efs.add(int(row["EF"]))
    return sorted(efs)


def pop_variance(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def stats(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    var = pop_variance(values)
    return {
        "n": n,
        "mean": mean,
        "variance": var,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
    }


def pct_change(old: float, new: float) -> float:
    if old == 0:
        return float("nan")
    return (new - old) / old * 100.0


# ── display ───────────────────────────────────────────────────────────────────

def print_comparison(dataset: str, baseline_ef: int,
                     base: dict, shiro: dict) -> None:
    var_delta = pct_change(base["variance"], shiro["variance"])
    std_delta = pct_change(base["std"],      shiro["std"])
    mean_delta = pct_change(base["mean"],    shiro["mean"])

    def sign(v: float) -> str:
        if math.isnan(v): return "n/a"
        return f"{v:+.2f}%"

    arrow = "▼" if var_delta < 0 else ("▲" if var_delta > 0 else "─")

    W = 60
    print()
    print("=" * W)
    print(f"  Dataset : {dataset}")
    print(f"  Baseline EF : {baseline_ef}   |   Shiro: adaptive per-query EF")
    print("=" * W)
    print(f"  {'Metric':<22} {'Baseline':>12} {'Shiro-EF':>12} {'Δ':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
    rows = [
        ("Queries (n)",    f"{base['n']:>12,}",  f"{shiro['n']:>12,}",  ""),
        ("Recall mean",    f"{base['mean']:>12.6f}", f"{shiro['mean']:>12.6f}", sign(mean_delta)),
        ("Recall variance",f"{base['variance']:>12.2e}", f"{shiro['variance']:>12.2e}", sign(var_delta)),
        ("Recall std",     f"{base['std']:>12.6f}", f"{shiro['std']:>12.6f}",  sign(std_delta)),
        ("Recall min",     f"{base['min']:>12.4f}", f"{shiro['min']:>12.4f}",  ""),
        ("Recall max",     f"{base['max']:>12.4f}", f"{shiro['max']:>12.4f}",  ""),
    ]
    for label, bval, sval, delta in rows:
        print(f"  {label:<22} {bval} {sval} {delta:>10}")
    print("=" * W)
    print(f"  Variance change: {arrow}  {sign(var_delta)}")
    print("=" * W)
    print()


def write_csv_result(dataset: str, baseline_ef: int,
                     base: dict, shiro: dict, out_dir: Path) -> None:
    var_delta  = pct_change(base["variance"], shiro["variance"])
    std_delta  = pct_change(base["std"],      shiro["std"])
    mean_delta = pct_change(base["mean"],     shiro["mean"])

    out = out_dir / f"variance_compare_{dataset}_ef{baseline_ef}.csv"
    fields = ["metric", "baseline", "shiro_ef", "pct_change"]
    rows = [
        ("n_queries",       base["n"],        shiro["n"],        ""),
        ("recall_mean",     base["mean"],      shiro["mean"],     f"{mean_delta:.4f}"),
        ("recall_variance", base["variance"],  shiro["variance"], f"{var_delta:.4f}"),
        ("recall_std",      base["std"],       shiro["std"],      f"{std_delta:.4f}"),
        ("recall_min",      base["min"],       shiro["min"],      ""),
        ("recall_max",      base["max"],       shiro["max"],      ""),
    ]
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow(r)
    print(f"  → saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare recall variance: baseline fixed-EF vs Shiro adaptive-EF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", nargs="?", help="Dataset slug (e.g. deep-image-96-angular)")
    parser.add_argument("baseline_ef", nargs="?", type=int, help="EF value to use from baseline CSV")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    args = parser.parse_args()

    datasets = available_datasets()

    if args.list or args.dataset is None:
        print("\nAvailable datasets:")
        for d in datasets:
            p = CSV_DIR / f"per_query_baseline_{d}.csv"
            efs = available_efs(p)
            print(f"  {d}  (EFs: {efs})")
        print()
        if args.dataset is None and not args.list:
            parser.print_help()
        sys.exit(0)

    dataset = args.dataset
    if dataset not in datasets:
        print(f"[ERROR] Unknown dataset '{dataset}'.")
        print(f"        Run with --list to see available options.")
        sys.exit(1)

    baseline_path = CSV_DIR / f"per_query_baseline_{dataset}.csv"
    results_path  = CSV_DIR / f"per_query_results_{dataset}.csv"

    if not results_path.exists():
        print(f"[ERROR] No shiro results file found: {results_path}")
        sys.exit(1)

    if args.baseline_ef is None:
        efs = available_efs(baseline_path)
        print(f"\n[ERROR] baseline_ef is required.")
        print(f"        Available EFs for '{dataset}': {efs}")
        sys.exit(1)

    baseline_ef = args.baseline_ef
    base_recalls = load_recalls_at_ef(baseline_path, baseline_ef)
    if not base_recalls:
        efs = available_efs(baseline_path)
        print(f"[ERROR] EF={baseline_ef} not found in baseline.")
        print(f"        Available EFs: {efs}")
        sys.exit(1)

    shiro_recalls = load_all_recalls(results_path)

    base_stats  = stats(base_recalls)
    shiro_stats = stats(shiro_recalls)

    print_comparison(dataset, baseline_ef, base_stats, shiro_stats)
    write_csv_result(dataset, baseline_ef, base_stats, shiro_stats, CSV_DIR)


if __name__ == "__main__":
    main()
