import os
import argparse
import numpy as np
import pandas as pd

def generate_summary(match_queries=False):
    csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"
    output_file = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv/summary_metrics.csv"
    if match_queries:
        print("Running in MATCHED QUERIES mode: SHIRO-EF percentiles will use the exact Query IDs from the baseline's percentiles.")
    else:
        print("Running in INDEPENDENT mode: Percentiles calculated separately for baseline and SHIRO-EF.")

    rows = []

    print(f"Scanning directory: {csv_dir}")

    datasets = set()
    for file in os.listdir(csv_dir):
        if file.startswith("per_query_baseline_") and file.endswith(".csv"):
            datasets.add(file.replace("per_query_baseline_", "").replace(".csv", ""))
        elif file.startswith("per_query_results_") and file.endswith(".csv"):
            datasets.add(file.replace("per_query_results_", "").replace(".csv", ""))

    for ds in datasets:
        print(f"Processing dataset: {ds}")

        base_file = os.path.join(csv_dir, f"per_query_baseline_{ds}.csv")
        mine_file = os.path.join(csv_dir, f"per_query_results_{ds}.csv")
        
        df_base = None
        df_mine = None
        
        if os.path.exists(base_file):
            df_base = pd.read_csv(base_file)
            
        if os.path.exists(mine_file):
            df_mine = pd.read_csv(mine_file)

        # 1. Process Baseline CSV
        if df_base is not None:
            for ef, group in df_base.groupby("EF"):
                avg_rec = group["Recall"].mean()
                avg_lat = int(round(group["Latency(ns)"].mean()))

                p05_rec = group["Recall"].quantile(0.05, interpolation="nearest")
                p01_rec = group["Recall"].quantile(0.01, interpolation="nearest")

                lat_5th_rec = int(
                    round(
                        group[np.isclose(group["Recall"], p05_rec)][
                            "Latency(ns)"
                        ].mean()
                    )
                )
                lat_1st_rec = int(
                    round(
                        group[np.isclose(group["Recall"], p01_rec)][
                            "Latency(ns)"
                        ].mean()
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
        if df_mine is not None:
            avg_rec = df_mine["Recall"].mean()
            avg_lat = int(round(df_mine["Latency(ns)"].mean()))

            if "EF" in df_mine.columns:
                adaptive_ef = df_mine["EF"].iloc[0]
            else:
                adaptive_ef = "WAE"

            if match_queries and df_base is not None:
                # Find the closest EF in the baseline if exact doesn't exist
                unique_efs = df_base["EF"].unique()
                if adaptive_ef == "WAE":
                    closest_ef = unique_efs[-1] # fallback to max
                else:
                    closest_ef = unique_efs[np.argmin(np.abs(unique_efs - adaptive_ef))]
                
                print(f"  [Matched Queries] SHIRO-EF={adaptive_ef}. Using baseline queries at EF={closest_ef}")
                
                base_at_ef = df_base[df_base["EF"] == closest_ef]
                
                base_p05_rec = base_at_ef["Recall"].quantile(0.05, interpolation="nearest")
                base_p01_rec = base_at_ef["Recall"].quantile(0.01, interpolation="nearest")
                
                qids_p05 = base_at_ef[np.isclose(base_at_ef["Recall"], base_p05_rec)]["QueryID"]
                qids_p01 = base_at_ef[np.isclose(base_at_ef["Recall"], base_p01_rec)]["QueryID"]
                
                p05_rec = df_mine[df_mine["QueryID"].isin(qids_p05)]["Recall"].mean()
                p01_rec = df_mine[df_mine["QueryID"].isin(qids_p01)]["Recall"].mean()
                
                lat_5th_rec = int(round(df_mine[df_mine["QueryID"].isin(qids_p05)]["Latency(ns)"].mean()))
                lat_1st_rec = int(round(df_mine[df_mine["QueryID"].isin(qids_p01)]["Latency(ns)"].mean()))
            else:
                # Independent mode (Original logic)
                p05_rec = df_mine["Recall"].quantile(0.05, interpolation="nearest")
                p01_rec = df_mine["Recall"].quantile(0.01, interpolation="nearest")
                lat_5th_rec = int(round(df_mine[np.isclose(df_mine["Recall"], p05_rec)]["Latency(ns)"].mean()))
                lat_1st_rec = int(round(df_mine[np.isclose(df_mine["Recall"], p01_rec)]["Latency(ns)"].mean()))

            rows.append(
                {
                    "dataset": ds,
                    "method": "adaptive",
                    "ef": adaptive_ef,
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

    out_df["ef_sort"] = pd.to_numeric(out_df["ef"], errors="coerce")
    out_df["ef_sort"] = out_df["ef_sort"].fillna(0)
    out_df.sort_values(
        ["dataset", "method", "ef_sort"], ascending=[True, True, True], inplace=True
    )
    out_df.drop(columns=["ef_sort"], inplace=True)

    float_cols = ["avg_recall", "5th_perc_recall", "1st_perc_recall"]
    out_df[float_cols] = out_df[float_cols].round(5)

    out_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully generated summary CSV at: {output_file}")

    print("\nPreview of summary_metrics.csv:")
    print(out_df.head(15).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary metrics CSV.")
    parser.add_argument(
        "--match-queries",
        action="store_true",
        help="Use exact query IDs from baseline percentiles to calculate SHIRO-EF percentiles."
    )
    args = parser.parse_args()
    
    generate_summary(match_queries=args.match_queries)
