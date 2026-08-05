import argparse
import os

import numpy as np
import pandas as pd


def _worst_query_ids(group: pd.DataFrame, percentile: float) -> pd.Index:
    threshold = group["Recall"].quantile(percentile, interpolation="nearest")
    return group.loc[group["Recall"] <= threshold, "QueryID"]


def generate_summary() -> None:
    csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro"
    output_file = (
        "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro/summary_metrics.csv"
    )

    print(
        "Percentile sets are always derived from the baseline. "
        "Both baseline and adaptive metrics are evaluated on those same query IDs."
    )

    rows = []
    print(f"Scanning directory: {csv_dir}")

    datasets: set[str] = set()
    for file in os.listdir(csv_dir):
        if file.startswith("per_query_baseline_") and file.endswith(".csv"):
            datasets.add(file.removeprefix("per_query_baseline_").removesuffix(".csv"))
        elif file.startswith("per_query_results_") and file.endswith(".csv"):
            datasets.add(file.removeprefix("per_query_results_").removesuffix(".csv"))

    for ds in sorted(datasets):
        print(f"Processing dataset: {ds}")

        base_file = os.path.join(csv_dir, f"per_query_baseline_{ds}.csv")
        mine_file = os.path.join(csv_dir, f"per_query_results_{ds}.csv")

        df_base = pd.read_csv(base_file) if os.path.exists(base_file) else None
        df_mine = pd.read_csv(mine_file) if os.path.exists(mine_file) else None

        if df_base is not None:
            for ef, group in df_base.groupby("EF"):
                avg_rec = group["Recall"].mean()
                avg_lat = int(round(group["Latency(ns)"].mean()))

                qids_p05 = _worst_query_ids(group, 0.05)
                qids_p01 = _worst_query_ids(group, 0.01)

                p05_rec = group.loc[group["QueryID"].isin(qids_p05), "Recall"].mean()
                p01_rec = group.loc[group["QueryID"].isin(qids_p01), "Recall"].mean()
                lat_5th = int(
                    round(
                        group.loc[group["QueryID"].isin(qids_p05), "Latency(ns)"].mean()
                    )
                )
                lat_1st = int(
                    round(
                        group.loc[group["QueryID"].isin(qids_p01), "Latency(ns)"].mean()
                    )
                )

                rows.append(
                    {
                        "dataset": ds,
                        "method": "baseline",
                        "ef": ef,
                        "avg_recall": avg_rec,
                        "5th_perc_recall": p05_rec,
                        "1st_perc_recall": p01_rec,
                        "avg_lat(ns)": avg_lat,
                        "lat_5th_rec": lat_5th,
                        "lat_1st_rec": lat_1st,
                    }
                )

        if df_mine is not None:
            avg_rec = df_mine["Recall"].mean()
            avg_lat = int(round(df_mine["Latency(ns)"].mean()))

            adaptive_ef = df_mine["EF"].iloc[0] if "EF" in df_mine.columns else "WAE"

            if df_base is not None:
                unique_efs = df_base["EF"].unique()
                if adaptive_ef == "WAE":
                    closest_ef = unique_efs[-1]
                else:
                    closest_ef = unique_efs[np.argmin(np.abs(unique_efs - adaptive_ef))]

                print(
                    f"  Adaptive EF={adaptive_ef} → using baseline EF={closest_ef} "
                    f"to define worst-5%/1% query sets"
                )

                base_at_ef = df_base[df_base["EF"] == closest_ef]
                qids_p05 = _worst_query_ids(base_at_ef, 0.05)
                qids_p01 = _worst_query_ids(base_at_ef, 0.01)

                p05_rec = df_mine.loc[
                    df_mine["QueryID"].isin(qids_p05), "Recall"
                ].mean()
                p01_rec = df_mine.loc[
                    df_mine["QueryID"].isin(qids_p01), "Recall"
                ].mean()
                lat_5th = int(
                    round(
                        df_mine.loc[
                            df_mine["QueryID"].isin(qids_p05), "Latency(ns)"
                        ].mean()
                    )
                )
                lat_1st = int(
                    round(
                        df_mine.loc[
                            df_mine["QueryID"].isin(qids_p01), "Latency(ns)"
                        ].mean()
                    )
                )
            else:
                # No baseline available — fall back to self-derived percentiles.
                print(
                    f"  WARNING: no baseline found for {ds}; "
                    f"using adaptive-only percentile sets (no cross-method comparison)"
                )
                qids_p05 = _worst_query_ids(df_mine, 0.05)
                qids_p01 = _worst_query_ids(df_mine, 0.01)

                p05_rec = df_mine.loc[
                    df_mine["QueryID"].isin(qids_p05), "Recall"
                ].mean()
                p01_rec = df_mine.loc[
                    df_mine["QueryID"].isin(qids_p01), "Recall"
                ].mean()
                lat_5th = int(
                    round(
                        df_mine.loc[
                            df_mine["QueryID"].isin(qids_p05), "Latency(ns)"
                        ].mean()
                    )
                )
                lat_1st = int(
                    round(
                        df_mine.loc[
                            df_mine["QueryID"].isin(qids_p01), "Latency(ns)"
                        ].mean()
                    )
                )

            rows.append(
                {
                    "dataset": ds,
                    "method": "adaptive",
                    "ef": adaptive_ef,
                    "avg_recall": avg_rec,
                    "5th_perc_recall": p05_rec,
                    "1st_perc_recall": p01_rec,
                    "avg_lat(ns)": avg_lat,
                    "lat_5th_rec": lat_5th,
                    "lat_1st_rec": lat_1st,
                }
            )

    out_df = pd.DataFrame(rows)

    out_df["ef_sort"] = pd.to_numeric(out_df["ef"], errors="coerce").fillna(0)
    out_df.sort_values(
        ["dataset", "method", "ef_sort"], ascending=[True, True, True], inplace=True
    )
    out_df.drop(columns=["ef_sort"], inplace=True)

    out_df[["avg_recall", "5th_perc_recall", "1st_perc_recall"]] = out_df[
        ["avg_recall", "5th_perc_recall", "1st_perc_recall"]
    ].round(5)

    out_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully generated summary CSV at: {output_file}")
    print("\nPreview of summary_metrics.csv:")
    print(out_df.head(15).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary metrics CSV.")
    # --match-queries is kept for backward CLI compatibility but is now a no-op;
    # baseline-derived query sets are always used.
    parser.add_argument(
        "--match-queries",
        action="store_true",
        help="(no-op, kept for compatibility) Query sets are always derived from the baseline.",
    )
    parser.parse_args()
    generate_summary()
