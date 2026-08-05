#!/usr/bin/env python3
"""
Aggregate per-query attempt CSVs (rep0/rep1/rep2) into a single CSV.

Input directory structure:
  <attempts_dir>/per_query_{kind}_{dataset}_rep{N}.csv

Output directory:
  <output_dir>/per_query_{kind}_{dataset}.csv

Aggregation strategy (--agg):
  min    - minimum latency across reps  [default]
  median - median latency across reps
  max    - maximum latency across reps
  file-median - calculate avg latency for each file, output the file that has the median avg latency

Usage examples:
  python aggregate_attempts.py
  python aggregate_attempts.py --agg median
  python aggregate_attempts.py --attempts-dir /path/to/attempts --output-dir /path/to/out
  python aggregate_attempts.py --kind baseline --agg max
  python aggregate_attempts.py --agg file-median
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-query attempt CSVs -> single CSV per dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--attempts-dir",
        type=Path,
        default=Path(__file__).parent / "csv_ada" / "attempts",
        help="Directory containing per_query_*_rep*.csv files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "csv_ada",
        help="Directory to write aggregated CSVs",
    )
    p.add_argument(
        "--agg",
        choices=["min", "median", "max", "file-median"],
        default="min",
        help="Latency aggregation strategy (default: min)",
    )
    p.add_argument(
        "--kind",
        choices=["results", "baseline", "both"],
        default="both",
        help="Which file kind to aggregate (default: both)",
    )
    return p.parse_args()


LATENCY_COL = "Latency(ns)"
QUERYID_COL = "QueryID"
EF_COL = "EF"
RECALL_COL = "Recall"
FIELDNAMES = [QUERYID_COL, EF_COL, LATENCY_COL, RECALL_COL]


def aggregate_latency(values: list[float], strategy: str) -> float:
    if strategy == "min":
        return min(values)
    if strategy == "max":
        return max(values)
    return statistics.median(values)


def collect_rep_files(attempts_dir: Path, kind: str) -> dict[str, list[Path]]:
    prefix = f"per_query_{kind}_"
    groups: dict[str, list[Path]] = defaultdict(list)

    for f in attempts_dir.iterdir():
        if not f.name.endswith(".csv"):
            continue
        if not f.name.startswith(prefix):
            continue
        stem = f.name[len(prefix) : -4]
        idx = stem.rfind("_rep")
        if idx == -1:
            continue
        dataset = stem[:idx]
        groups[dataset].append(f)

    for dataset in groups:
        groups[dataset].sort(key=lambda p: p.stem)

    return dict(groups)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def aggregate(rep_files: list[Path], strategy: str) -> list[dict]:
    if strategy == "file-median":
        file_means = []
        file_rows = []
        for path in rep_files:
            rows = read_csv_rows(path)
            if not rows:
                file_means.append(0.0)
                file_rows.append(rows)
                continue
            mean_lat = sum(float(r[LATENCY_COL]) for r in rows) / len(rows)
            file_means.append(mean_lat)
            file_rows.append(rows)

        median_mean = statistics.median(file_means)
        idx = file_means.index(median_mean)

        sorted_keys = sorted(
            file_rows[idx], key=lambda x: (int(x[EF_COL]), int(x[QUERYID_COL]))
        )
        return sorted_keys

    latencies: dict[tuple[str, str], list[float]] = defaultdict(list)
    representative: dict[tuple[str, str], dict] = {}

    for path in rep_files:
        for row in read_csv_rows(path):
            key = (row[QUERYID_COL], row[EF_COL])
            latencies[key].append(float(row[LATENCY_COL]))
            if key not in representative:
                representative[key] = row

    sorted_keys = sorted(representative.keys(), key=lambda x: (int(x[1]), int(x[0])))

    result = []
    for key in sorted_keys:
        chosen_lat = aggregate_latency(latencies[key], strategy)
        ref = representative[key]
        result.append(
            {
                QUERYID_COL: key[0],
                EF_COL: key[1],
                LATENCY_COL: int(chosen_lat),
                RECALL_COL: ref[RECALL_COL],
            }
        )
    return result


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    attempts_dir: Path = args.attempts_dir
    output_dir: Path = args.output_dir
    strategy: str = args.agg
    kind_arg: str = args.kind

    if not attempts_dir.is_dir():
        sys.exit(f"Error: attempts directory not found: {attempts_dir}")

    kinds = ["results", "baseline"] if kind_arg == "both" else [kind_arg]

    any_processed = False
    for kind in kinds:
        groups = collect_rep_files(attempts_dir, kind)
        if not groups:
            print(f"[{kind}] No rep files found in {attempts_dir} - skipping.")
            continue

        for dataset, rep_files in sorted(groups.items()):
            print(
                f"[{kind}] {dataset}: merging {len(rep_files)} rep(s) with agg={strategy} ...",
                end=" ",
            )
            rows = aggregate(rep_files, strategy)
            out_path = output_dir / f"per_query_{kind}_{dataset}.csv"
            write_csv(rows, out_path)
            print(f"-> {out_path}  ({len(rows)} rows)")
            any_processed = True

    if not any_processed:
        sys.exit("No files were processed. Check --attempts-dir and --kind.")

    print(f"\nDone. Aggregation strategy: {strategy}")


if __name__ == "__main__":
    main()
