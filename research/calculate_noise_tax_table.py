import os

import pandas as pd

csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"

results = []

for file in sorted(os.listdir(csv_dir)):
    if file.startswith("per_query_baseline_") and file.endswith(".csv"):
        dataset_name = file.replace("per_query_baseline_", "").replace(".csv", "")
        base_csv = os.path.join(csv_dir, file)

        df_base = pd.read_csv(base_csv)

        unique_efs = sorted(df_base["EF"].unique())

        for ef_star in unique_efs:
            # Filter baseline for EFs <= ef_star to find the minimum sufficient EF
            df_base_filtered = df_base[df_base["EF"] <= ef_star].copy()

            # Get target recall and latency for each query at ef
            df_star = df_base_filtered[df_base_filtered["EF"] == ef_star][
                ["QueryID", "Recall", "Latency(ns)"]
            ].copy()
            df_star.rename(
                columns={"Recall": "Recall_star", "Latency(ns)": "Latency_star"},
                inplace=True,
            )

            # Merge target recall back to the filtered dataframe
            df_base_filtered = df_base_filtered.merge(
                df_star[["QueryID", "Recall_star"]], on="QueryID"
            )

            # Find minimum EF where Recall >= Recall_star
            df_sufficient = df_base_filtered[
                df_base_filtered["Recall"] >= df_base_filtered["Recall_star"]
            ]

            # For each query, find the row with the minimum EF
            idx_min_ef = df_sufficient.groupby("QueryID")["EF"].idxmin()
            df_ef_i = df_sufficient.loc[idx_min_ef][["QueryID", "EF", "Latency(ns)"]]
            df_ef_i.rename(
                columns={"EF": "ef_i_star", "Latency(ns)": "Latency_i_star"},
                inplace=True,
            )

            # Merge back to calculate Noise Tax
            df_tax = df_star.merge(df_ef_i, on="QueryID")

            df_tax["NoiseTax(ns)"] = df_tax["Latency_star"] - df_tax["Latency_i_star"]
            total_noise_tax_ms = df_tax["NoiseTax(ns)"].sum() / 1e6
            total_latency_star_ms = df_tax["Latency_star"].sum() / 1e6

            percent_tax = (
                (total_noise_tax_ms / total_latency_star_ms) * 100
                if total_latency_star_ms > 0
                else 0
            )

            results.append(
                {
                    "Dataset": dataset_name,
                    "Global EF (ef)": ef_star,
                    "Total Latency (ms)": total_latency_star_ms,
                    "Noise Tax (ms)": total_noise_tax_ms,
                    "Noise Tax %": percent_tax,
                }
            )

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(csv_dir, "noise_tax_table_all_efs.csv"), index=False)
