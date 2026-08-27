#!/usr/bin/env python3
"""
recall_variance.py — Compute recall variance across queries for each EF level.

Usage:
    python recall_variance.py <csv_file> [<csv_file2> ...]
    python recall_variance.py research/csv_shiro/per_query_baseline_deep-image-96-angular.csv

Output columns (per EF bucket):
    EF, n_queries, recall_mean, recall_variance, recall_std, recall_min, recall_max
"""

import sys
import csv
import math
from pathlib import Path
from collections import defaultdict


def load_recalls(path: Path) -> dict[int, list[float]]:
    """Return {EF: [recall, ...]} from a per-query CSV."""
    ef_recalls: dict[int, list[float]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ef = int(row["EF"])
            recall = float(row["Recall"])
            ef_recalls[ef].append(recall)
    return dict(ef_recalls)


def variance(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n          # population variance


def summarise(ef: int, recalls: list[float]) -> dict:
    n = len(recalls)
    mean = sum(recalls) / n
    var = variance(recalls)
    return {
        "EF": ef,
        "n_queries": n,
        "recall_mean": mean,
        "recall_variance": var,
        "recall_std": math.sqrt(var),
        "recall_min": min(recalls),
        "recall_max": max(recalls),
    }


def print_table(rows: list[dict], title: str) -> None:
    header = ["EF", "n_queries", "recall_mean", "recall_variance", "recall_std", "recall_min", "recall_max"]
    col_w: dict[str, int] = {
        h: max(len(h), max(len(f"{r[h]:.6g}") if isinstance(r[h], float) else len(str(r[h])) for r in rows))
        for h in header
    }

    sep = "+-" + "-+-".join("-" * w for w in col_w.values()) + "-+"
    fmt_row = "| " + " | ".join(
        f"{{:{col_w[h]}}}" if h in ("EF", "n_queries") else f"{{:{col_w[h]}.6g}}"
        for h in header
    ) + " |"

    print(f"\n{'='*len(sep)}")
    print(f"  {title}")
    print(sep)
    print("| " + " | ".join(f"{h:{col_w[h]}}" for h in header) + " |")
    print(sep)
    for r in rows:
        vals = [r[h] for h in header]
        line = "| "
        parts = []
        for h, v in zip(header, vals):
            if isinstance(v, float):
                parts.append(f"{v:{col_w[h]}.6g}")
            else:
                parts.append(f"{v:{col_w[h]}}")
        print("| " + " | ".join(parts) + " |")
    print(sep)


def write_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["EF", "n_queries", "recall_mean", "recall_variance", "recall_std", "recall_min", "recall_max"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → saved: {out_path}")


def process(path: Path) -> None:
    ef_recalls = load_recalls(path)
    rows = [summarise(ef, recalls) for ef, recalls in sorted(ef_recalls.items())]
    print_table(rows, f"{path.name}")
    out = path.parent / (path.stem + "_recall_variance.csv")
    write_csv(rows, out)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = [Path(a) for a in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            print(f"[ERROR] file not found: {p}", file=sys.stderr)
            sys.exit(1)
        process(p)


if __name__ == "__main__":
    main()
