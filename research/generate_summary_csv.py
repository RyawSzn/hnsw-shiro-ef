import os

import numpy as np
import pandas as pd


def generate_summary():
    csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"
    output_file = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv/summary_metrics.csv"

    rows = []

    print(f"Scanning directory: {csv_dir}")

    # Identify all unique datasets by scanning filenames
    datasets = set()
    for file in os.listdir(csv_dir):
        if file.startswith("per_query_baseline_") and file.endswith(".csv"):
            datasets.add(file.replace("per_query_baseline_", "").replace(".csv", ""))
        elif file.startswith("per_query_results_") and file.endswith(".csv"):
            datasets.add(file.replace("per_query_results_", "").replace(".csv", ""))

    for ds in datasets:
        print(f"Processing dataset: {ds}")

        # 1. Process Baseline CSV (contains multiple EF parameters)
        base_file = os.path.join(csv_dir, f"per_query_baseline_{ds}.csv")
        if os.path.exists(base_file):
            df_base = pd.read_csv(base_file)

            # Group by EF because baseline has sweeps (e.g., EF=10, EF=50, etc.)
            for ef, group in df_base.groupby("EF"):
                avg_rec = group["Recall"].mean()
                avg_lat = int(round(group["Latency(ns)"].mean()))

                # Calculate the "At" metric (Point Estimate).
                # Recall is the exact 5th/1st percentile value (using 'nearest' to ensure it's an exact dataset value)
                p05_rec = group["Recall"].quantile(0.05, interpolation="nearest")
                p01_rec = group["Recall"].quantile(0.01, interpolation="nearest")

                # Pull the exact subset of queries that achieved this specific recall and take their median latency
                # (using np.isclose to safely match floating point numbers)
                lat_5th_rec = int(
                    round(
                        group[np.isclose(group["Recall"], p05_rec)][
                            "Latency(ns)"
                        ].median()
                    )
                )
                lat_1st_rec = int(
                    round(
                        group[np.isclose(group["Recall"], p01_rec)][
                            "Latency(ns)"
                        ].median()
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
                        "lat_5th_rec": lat_5th_rec,
                        "lat_1st_rec": lat_1st_rec,
                    }
                )

        # 2. Process Adaptive / shiro-ef CSV
        mine_file = os.path.join(csv_dir, f"per_query_results_{ds}.csv")
        if os.path.exists(mine_file):
            df_mine = pd.read_csv(mine_file)

            avg_rec = df_mine["Recall"].mean()
            avg_lat = int(round(df_mine["Latency(ns)"].mean()))

            # Calculate the "At" metrics for Adaptive
            p05_rec = df_mine["Recall"].quantile(0.05, interpolation="nearest")
            p01_rec = df_mine["Recall"].quantile(0.01, interpolation="nearest")

            # Pull the exact subset of queries that achieved this specific recall and take their mean latency
            lat_5th_rec = int(
                round(
                    df_mine[np.isclose(df_mine["Recall"], p05_rec)][
                        "Latency(ns)"
                    ].mean()
                )
            )
            lat_1st_rec = int(
                round(
                    df_mine[np.isclose(df_mine["Recall"], p01_rec)][
                        "Latency(ns)"
                    ].mean()
                )
            )

            # Use EF if present in the results, otherwise fallback to "WAE"
            if "EF" in df_mine.columns:
                adaptive_ef = df_mine["EF"].iloc[0]
            else:
                adaptive_ef = "WAE"

            rows.append(
                {
                    "dataset": ds,
                    "method": "adaptive",
                    "ef": adaptive_ef,  # Represent adaptive EF dynamically or WAE
                    "avg_recall": avg_rec,
                    "5th_perc_recall": p05_rec,
                    "1st_perc_recall": p01_rec,
                    "avg_lat(ns)": avg_lat,
                    "lat_5th_rec": lat_5th_rec,
                    "lat_1st_rec": lat_1st_rec,
                }
            )

    # Create DataFrame and sort for clean reading
    out_df = pd.DataFrame(rows)

    # Sort by Dataset -> Method (adaptive first, then baseline) -> EF value
    # Use a safe sorting column for EF
    out_df["ef_sort"] = pd.to_numeric(out_df["ef"], errors="coerce")
    out_df["ef_sort"] = out_df["ef_sort"].fillna(0)
    out_df.sort_values(
        ["dataset", "method", "ef_sort"], ascending=[True, True, True], inplace=True
    )
    out_df.drop(columns=["ef_sort"], inplace=True)

    # Round floats for cleaner CSV output
    float_cols = ["avg_recall", "5th_perc_recall", "1st_perc_recall"]
    out_df[float_cols] = out_df[float_cols].round(5)

    # Save to CSV
    out_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully generated summary CSV at: {output_file}")

    # Print a quick preview of the results
    print("\nPreview of summary_metrics.csv:")
    print(out_df.head(15).to_string())


if __name__ == "__main__":
    generate_summary()
