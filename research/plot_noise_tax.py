import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"
img_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story"
os.makedirs(img_dir, exist_ok=True)

csv_path = os.path.join(csv_dir, "noise_tax_table_all_efs.csv")
if not os.path.exists(csv_path):
    print("Run calculate_noise_tax_table.py first.")
    exit(1)

df = pd.read_csv(csv_path)

# 1. Calculate Adaptive Noise Tax
adaptive_metrics = {}
for dataset_name in df["Dataset"].unique():
    mine_csv = os.path.join(csv_dir, f"per_query_results_{dataset_name}.csv")
    base_csv = os.path.join(csv_dir, f"per_query_baseline_{dataset_name}.csv")
    
    if not os.path.exists(mine_csv) or not os.path.exists(base_csv):
        continue
        
    df_mine = pd.read_csv(mine_csv)
    df_base = pd.read_csv(base_csv)
    
    merged = df_base.merge(df_mine[["QueryID", "Recall", "Latency(ns)"]], on="QueryID", suffixes=("_base", "_mine"))
    merged.sort_values(["QueryID", "EF"], inplace=True)
    
    def get_min_ef_latency(grp):
        # Find minimum baseline EF that achieves AT LEAST the recall the adaptive method achieved
        suff = grp[grp["Recall_base"] >= grp["Recall_mine"]]
        if not suff.empty:
            return suff.iloc[0]["Latency(ns)_base"]
        else:
            return grp.iloc[-1]["Latency(ns)_base"]
            
    essential_lat_ns = merged.groupby("QueryID").apply(get_min_ef_latency)
    
    total_mine_lat_ms = df_mine["Latency(ns)"].sum() / 1e6
    total_essential_ms = essential_lat_ns.sum() / 1e6
    
    # Adaptive noise tax is how much extra it spent over the optimal baseline
    noise_tax_ms = max(0, total_mine_lat_ms - total_essential_ms)
    noise_tax_pct = (noise_tax_ms / total_mine_lat_ms) * 100 if total_mine_lat_ms > 0 else 0
    
    adaptive_metrics[dataset_name] = {
        "total_lat_ms": total_mine_lat_ms,
        "essential_ms": total_essential_ms,
        "noise_tax_ms": noise_tax_ms,
        "noise_tax_pct": noise_tax_pct
    }

# 2. Combined Line Plot for Noise Tax Percentage
plt.figure(figsize=(14, 9))
plt.style.use("seaborn-v0_8-whitegrid")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, dataset in enumerate(df["Dataset"].unique()):
    if dataset not in adaptive_metrics: continue
    
    df_ds = df[df["Dataset"] == dataset].copy()
    df_ds.sort_values("Global EF (ef*)", inplace=True)
    
    c = colors[i % len(colors)]
    
    # Plot Baseline Noise Tax
    plt.plot(df_ds["Global EF (ef*)"], df_ds["Noise Tax %"], marker='o', linewidth=2, color=c, label=f"{dataset} (Static EF)")
    
    # Plot Adaptive Noise Tax as a horizontal line
    adp_tax = adaptive_metrics[dataset]["noise_tax_pct"]
    plt.axhline(y=adp_tax, color=c, linestyle='--', linewidth=2, alpha=0.8, label=f"{dataset} (SHIRO-EF)")

plt.title("Noise Tax Percentage vs. Global EF ($ef^*$) + SHIRO-EF Performance", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Global Safe $ef^*$ Configuration", fontsize=14, fontweight="bold")
plt.ylabel("Noise Tax (Wasted Compute) %", fontsize=14, fontweight="bold")
plt.legend(title="Datasets (Solid=Baseline, Dashed=SHIRO-EF)", fontsize=10, title_fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

combined_out = os.path.join(img_dir, "noise_tax_percentage_combined.png")
plt.tight_layout()
plt.savefig(combined_out, dpi=300)
print(f"Saved combined plot to: {combined_out}")
plt.close()

# 3. Individual Area Plots & Absolute Noise Plots
for dataset in df["Dataset"].unique():
    if dataset not in adaptive_metrics: continue
    
    df_ds = df[df["Dataset"] == dataset].copy()
    df_ds.sort_values("Global EF (ef*)", inplace=True)
    
    efs = df_ds["Global EF (ef*)"].values
    total_lat = df_ds["Total Latency (ms)"].values
    tax = df_ds["Noise Tax (ms)"].values
    essential_lat = total_lat - tax
    
    adp_total = adaptive_metrics[dataset]["total_lat_ms"]
    adp_tax_ms = adaptive_metrics[dataset]["noise_tax_ms"]
    adp_tax_pct = adaptive_metrics[dataset]["noise_tax_pct"]
    
    # --- PLOT A: Area Chart with Total Latency overlay ---
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    
    plt.fill_between(efs, 0, essential_lat, color="#2ecc71", alpha=0.5, label="Baseline Essential Compute")
    plt.fill_between(efs, essential_lat, total_lat, color="#e74c3c", alpha=0.5, label="Baseline Noise Tax (Wasted)")
    plt.plot(efs, total_lat, color="black", linewidth=2, label="Baseline Total Latency")
    
    # SHIRO-EF Overlay
    plt.axhline(y=adp_total, color='blue', linestyle='--', linewidth=2.5, label=f"SHIRO-EF Total Latency")
    
    plt.title(f"Compute Breakdown vs. Global EF: {dataset}", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Global Safe $ef^*$ Configuration", fontsize=14, fontweight="bold")
    plt.ylabel("Total Latency (ms)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=11)
    
    plt.tight_layout()
    ds_out = os.path.join(img_dir, f"noise_tax_area_{dataset}.png")
    plt.savefig(ds_out, dpi=300)
    print(f"Saved area plot to: {ds_out}")
    plt.close()
    
    # --- PLOT B: Absolute Noise Tax (ms) curve vs SHIRO-EF ---
    plt.figure(figsize=(10, 6))
    plt.plot(efs, tax, color="#e74c3c", linewidth=3, marker='o', label="Static EF Noise Tax (ms)")
    plt.axhline(y=adp_tax_ms, color="blue", linestyle='--', linewidth=3, label=f"SHIRO-EF Noise Tax (ms)")
    
    plt.fill_between(efs, adp_tax_ms, tax, where=(tax > adp_tax_ms), interpolate=True, color="#2ecc71", alpha=0.3, label="Compute Saved by SHIRO-EF")
    
    plt.title(f"Absolute Wasted Compute: {dataset}", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Global Safe $ef^*$ Configuration", fontsize=14, fontweight="bold")
    plt.ylabel("Noise Tax / Wasted Compute (ms)", fontsize=14, fontweight="bold")
    
    # Annotate SHIRO-EF
    plt.annotate(
        f"SHIRO-EF Waste:\n{adp_tax_pct:.1f}% ({adp_tax_ms:.1f} ms)",
        xy=(efs[len(efs)//2], adp_tax_ms),
        xytext=(efs[len(efs)//2], adp_tax_ms * 1.5 + max(tax)*0.1),
        arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
        fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9),
        ha="center"
    )

    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()
    abs_out = os.path.join(img_dir, f"noise_tax_absolute_{dataset}.png")
    plt.savefig(abs_out, dpi=300)
    print(f"Saved absolute noise plot to: {abs_out}")
    plt.close()

