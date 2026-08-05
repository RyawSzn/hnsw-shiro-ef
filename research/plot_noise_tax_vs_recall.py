import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"
img_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story"
os.makedirs(img_dir, exist_ok=True)

csv_path = os.path.join(csv_dir, "noise_tax_table_all_efs.csv")
if not os.path.exists(csv_path):
    print("Run calculate_noise_tax_table.py first.")
    exit(1)

df_tax = pd.read_csv(csv_path)

for dataset in df_tax["Dataset"].unique():
    mine_csv = os.path.join(csv_dir, f"per_query_results_{dataset}.csv")
    base_csv = os.path.join(csv_dir, f"per_query_baseline_{dataset}.csv")

    if not os.path.exists(mine_csv) or not os.path.exists(base_csv):
        continue

    df_mine = pd.read_csv(mine_csv)
    df_base = pd.read_csv(base_csv)

    # Calculate Mean Recall for Baseline at each EF
    base_recalls = df_base.groupby("EF")["Recall"].mean().reset_index()
    base_recalls.rename(columns={"EF": "Global EF (ef)"}, inplace=True)

    # Merge with noise tax table
    df_ds = df_tax[df_tax["Dataset"] == dataset].copy()
    df_ds = df_ds.merge(base_recalls, on="Global EF (ef)")
    df_ds.sort_values("Recall", inplace=True)

    # Calculate SHIRO-EF Noise Tax and Mean Recall
    merged = df_base.merge(
        df_mine[["QueryID", "Recall", "Latency(ns)"]],
        on="QueryID",
        suffixes=("_base", "_mine"),
    )
    merged.sort_values(["QueryID", "EF"], inplace=True)

    def get_min_ef_latency(grp):
        suff = grp[grp["Recall_base"] >= grp["Recall_mine"]]
        if not suff.empty:
            return suff.iloc[0]["Latency(ns)_base"]
        else:
            return grp.iloc[-1]["Latency(ns)_base"]

    essential_lat_ns = merged.groupby("QueryID").apply(get_min_ef_latency)

    total_mine_lat_ms = df_mine["Latency(ns)"].sum() / 1e6
    total_essential_ms = essential_lat_ns.sum() / 1e6

    adp_tax_ms = max(0, total_mine_lat_ms - total_essential_ms)
    adp_recall = df_mine["Recall"].mean()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot Baseline
    plt.plot(
        df_ds["Recall"],
        df_ds["Noise Tax (ms)"],
        marker="o",
        linewidth=3,
        color="#e74c3c",
        label="Static EF (Baseline)",
    )

    # Annotate EF values on points
    for _, row in df_ds.iterrows():
        plt.annotate(
            f"ef={int(row['Global EF (ef)'])}",
            (row["Recall"], row["Noise Tax (ms)"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # Plot SHIRO-EF
    plt.scatter(
        [adp_recall],
        [adp_tax_ms],
        color="blue",
        marker="*",
        s=300,
        zorder=5,
        label="SHIRO-EF",
    )

    # Highlight the gap vertically if there is a baseline point near the same recall
    # We can just draw lines to the axes
    plt.axvline(x=adp_recall, color="blue", linestyle=":", alpha=0.5)
    plt.axhline(y=adp_tax_ms, color="blue", linestyle=":", alpha=0.5)

    plt.title(
        f"Noise Tax vs. Recall: {dataset}", fontsize=16, fontweight="bold", pad=15
    )
    plt.xlabel("Mean Recall", fontsize=14, fontweight="bold")
    plt.ylabel("Noise Tax (Wasted Compute) in ms", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(img_dir, f"noise_tax_vs_recall_{dataset}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()
