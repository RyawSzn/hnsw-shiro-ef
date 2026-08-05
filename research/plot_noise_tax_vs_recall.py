import os

import matplotlib.pyplot as plt
import pandas as pd

CSV_SHIRO = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_shiro"
CSV_ADA   = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv_ada"
IMG_DIR   = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story_ada"
os.makedirs(IMG_DIR, exist_ok=True)

tax_path = os.path.join(CSV_SHIRO, "noise_tax_table_all_efs.csv")
if not os.path.exists(tax_path):
    print("Run calculate_noise_tax_table.py first.")
    exit(1)

df_tax = pd.read_csv(tax_path)


def adaptive_noise_tax(df_base, df_results):
    merged = df_base.merge(
        df_results[["QueryID", "Recall", "Latency(ns)"]],
        on="QueryID",
        suffixes=("_base", "_mine"),
    )
    merged.sort_values(["QueryID", "EF"], inplace=True)

    def min_ef_latency(grp):
        suff = grp[grp["Recall_base"] >= grp["Recall_mine"]]
        return suff.iloc[0]["Latency(ns)_base"] if not suff.empty else grp.iloc[-1]["Latency(ns)_base"]

    essential_ms = merged.groupby("QueryID").apply(min_ef_latency).sum() / 1e6
    total_ms     = df_results["Latency(ns)"].sum() / 1e6
    return max(0.0, total_ms - essential_ms), df_results["Recall"].mean()


for dataset in df_tax["Dataset"].unique():
    base_csv        = os.path.join(CSV_SHIRO, f"per_query_baseline_{dataset}.csv")
    shiro_mine_csv  = os.path.join(CSV_SHIRO, f"per_query_results_{dataset}.csv")
    ada_mine_csv    = os.path.join(CSV_ADA,   f"per_query_results_{dataset}.csv")

    if not os.path.exists(base_csv) or not os.path.exists(shiro_mine_csv):
        continue

    df_base      = pd.read_csv(base_csv)
    df_shiro     = pd.read_csv(shiro_mine_csv)

    base_recalls = df_base.groupby("EF")["Recall"].mean().reset_index()
    base_recalls.rename(columns={"EF": "Global EF (ef)"}, inplace=True)

    df_ds = df_tax[df_tax["Dataset"] == dataset].copy()
    df_ds = df_ds.merge(base_recalls, on="Global EF (ef)")
    df_ds.sort_values("Recall", inplace=True)

    shiro_tax_ms, shiro_recall = adaptive_noise_tax(df_base, df_shiro)

    has_ada = os.path.exists(ada_mine_csv)
    if has_ada:
        df_ada = pd.read_csv(ada_mine_csv)
        ada_tax_ms, ada_recall = adaptive_noise_tax(df_base, df_ada)

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.plot(
        df_ds["Recall"],
        df_ds["Noise Tax (ms)"],
        marker="o",
        linewidth=3,
        color="#e74c3c",
        label="Static EF (Baseline)",
    )

    for _, row in df_ds.iterrows():
        plt.annotate(
            f"ef={int(row['Global EF (ef)'])}",
            (row["Recall"], row["Noise Tax (ms)"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.scatter([shiro_recall], [shiro_tax_ms],
                color="blue", marker="*", s=300, zorder=5, label="SHIRO-EF")
    plt.axvline(x=shiro_recall, color="blue", linestyle=":", alpha=0.5)
    plt.axhline(y=shiro_tax_ms, color="blue", linestyle=":", alpha=0.5)

    if has_ada:
        plt.scatter([ada_recall], [ada_tax_ms],
                    color="darkorange", marker="x", s=200, linewidths=2.5,
                    zorder=5, label="Ada-EF")
        plt.axvline(x=ada_recall,  color="darkorange", linestyle=":", alpha=0.5)
        plt.axhline(y=ada_tax_ms,  color="darkorange", linestyle=":", alpha=0.5)

    plt.title(f"Noise Tax vs. Recall: {dataset}", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Mean Recall", fontsize=14, fontweight="bold")
    plt.ylabel("Noise Tax (Wasted Compute) in ms", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(IMG_DIR, f"noise_tax_vs_recall_{dataset}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.close()
