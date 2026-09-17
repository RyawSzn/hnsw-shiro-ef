#!/usr/bin/env python3
"""
apply_affine.py
───────────────
Apply per-dataset affine correction to per-query latencies.

Affine model (fitted on matched-EF pairs):
    Latency_adjusted = a_intercept_ns + b_slope * Latency_raw_ns

Input  : research/csv_ada/attempts/per_query_results_{dataset}_rep{N}.csv
Output : research/csv_ada/attempts_AFFINED/per_query_results_{dataset}_rep{N}.csv

Columns preserved: QueryID, EF, Latency(ns) [adjusted], Recall
"""

import csv
import re
import sys
from pathlib import Path

# ─── Affine parameters (from affine_fit_parameters.csv) ────────────────────
# key: dataset name (lowercase, as it appears in the filename)
# value: (a_intercept_ns, b_slope)
FIT_PARAMS: dict[str, tuple[float, float]] = {
    "cohere": (665792.0734755024, 1.0794752426412255),
    "deep-image-96-angular": (-1588.7612302524503, 1.02635614312979),
    "glove-100-angular": (-13547.639130830765, 1.0100890076767566),
    "msmarco": (-176938.1139998734, 1.133534998234818),
    "word2vec-300-angular": (-1889.4753057532944, 1.0046927031236035),
    # POOLED fallback (used when dataset not individually listed)
    "POOLED": (-227258.82271227986, 1.1110990044948736),
}

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ATTEMPTS_DIR = SCRIPT_DIR / "csv_ada" / "attempts_RAW"
OUT_DIR = SCRIPT_DIR / "csv_ada" / "attempts"

FILENAME_RE = re.compile(r"^per_query_results_(?P<dataset>.+?)_rep(?P<rep>\d+)\.csv$")


def get_params(dataset: str) -> tuple[float, float]:
    """Return (a, b) for dataset; fall back to POOLED if not found."""
    if dataset in FIT_PARAMS:
        return FIT_PARAMS[dataset]
    print(
        f"  [WARN] dataset '{dataset}' not in fit params — using POOLED",
        file=sys.stderr,
    )
    return FIT_PARAMS["POOLED"]


def adjust_latency(raw_ns: float, a: float, b: float) -> float:
    """Affine correction: adjusted = a + b * raw."""
    return a + b * raw_ns


def process_file(src: Path, dst: Path, a: float, b: float) -> int:
    """Read src, apply affine to Latency(ns), write dst. Returns row count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with src.open(newline="") as fin, dst.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Empty or headerless file: {src}")

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            raw_ns = float(row["Latency(ns)"])
            adj_ns = adjust_latency(raw_ns, a, b)
            row["Latency(ns)"] = f"{adj_ns:.6f}"
            writer.writerow(row)
            rows_written += 1

    return rows_written


def main() -> None:
    if not ATTEMPTS_DIR.exists():
        sys.exit(f"[ERROR] Input directory not found: {ATTEMPTS_DIR}")

    csv_files = sorted(ATTEMPTS_DIR.glob("per_query_results_*.csv"))
    if not csv_files:
        sys.exit(f"[ERROR] No matching CSV files in {ATTEMPTS_DIR}")

    print(f"Output directory : {OUT_DIR}")
    print(f"Files to process : {len(csv_files)}\n")

    total_rows = 0
    for src in csv_files:
        m = FILENAME_RE.match(src.name)
        if not m:
            print(f"  [SKIP] unrecognised filename: {src.name}", file=sys.stderr)
            continue

        dataset = m.group("dataset")
        rep = m.group("rep")
        a, b = get_params(dataset)

        dst = OUT_DIR / src.name
        rows = process_file(src, dst, a, b)
        total_rows += rows
        print(f"  {src.name}  →  a={a:+.2f}  b={b:.6f}  ({rows} rows)")

    print(f"\nDone. {len(csv_files)} files, {total_rows} total rows → {OUT_DIR}")


if __name__ == "__main__":
    main()
