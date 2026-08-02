import os
import pandas as pd
import matplotlib.pyplot as plt

csv_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv"
img_dir = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/img/story"
os.makedirs(img_dir, exist_ok=True)

csv_path = os.path.join(csv_dir, "noise_tax_table_all_efs.csv")
if not os.path.exists(csv_path):
    print("Run calculate_noise_tax_table.py first.")
    exit(1)

df = pd.read_csv(csv_path)

# 1. Combined Line Plot for Noise Tax Percentage
plt.figure(figsize=(12, 8))
plt.style.use("seaborn-v0_8-whitegrid")

for dataset in df["Dataset"].unique():
    df_ds = df[df["Dataset"] == dataset].copy()
    df_ds.sort_values("Global EF (ef*)", inplace=True)
    plt.plot(df_ds["Global EF (ef*)"], df_ds["Noise Tax %"], marker='o', linewidth=2, label=dataset)

plt.title("Noise Tax Percentage vs. Global EF ($ef^*$)", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Global Safe $ef^*$ Configuration", fontsize=14, fontweight="bold")
plt.ylabel("Noise Tax (Wasted Compute) %", fontsize=14, fontweight="bold")
plt.legend(title="Datasets", fontsize=11, title_fontsize=12, loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)

import matplotlib.ticker as mtick
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

combined_out = os.path.join(img_dir, "noise_tax_percentage_combined.png")
plt.tight_layout()
plt.savefig(combined_out, dpi=300)
print(f"Saved combined plot to: {combined_out}")
plt.close()

# 2. Individual Area Plots (Total Latency vs Essential Latency)
for dataset in df["Dataset"].unique():
    df_ds = df[df["Dataset"] == dataset].copy()
    df_ds.sort_values("Global EF (ef*)", inplace=True)
    
    efs = df_ds["Global EF (ef*)"].values
    total_lat = df_ds["Total Latency (ms)"].values
    tax = df_ds["Noise Tax (ms)"].values
    essential_lat = total_lat - tax
    
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    
    plt.fill_between(efs, 0, essential_lat, color="#2ecc71", alpha=0.6, label="Essential Compute")
    plt.fill_between(efs, essential_lat, total_lat, color="#e74c3c", alpha=0.6, label="Noise Tax (Wasted Compute)")
    plt.plot(efs, total_lat, color="black", linewidth=2, label="Total Latency (Static $ef^*$)")
    
    # Add a marker for the "optimal" ef if we know it (we can just plot the lines)
    plt.title(f"Noise Tax Breakdown: {dataset}", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Global Safe $ef^*$ Configuration", fontsize=14, fontweight="bold")
    plt.ylabel("Total Latency (ms)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=12)
    
    # Annotate max noise tax
    max_tax_idx = df_ds["Noise Tax %"].idxmax()
    max_tax_ef = df_ds.loc[max_tax_idx, "Global EF (ef*)"]
    max_tax_pct = df_ds.loc[max_tax_idx, "Noise Tax %"]
    
    plt.annotate(
        f"Max Wasted: {max_tax_pct:.1f}%",
        xy=(max_tax_ef, total_lat[df_ds.index == max_tax_idx][0]),
        xytext=(max_tax_ef * 0.8, total_lat[df_ds.index == max_tax_idx][0] * 1.1),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
    )
    
    plt.tight_layout()
    ds_out = os.path.join(img_dir, f"noise_tax_area_{dataset}.png")
    plt.savefig(ds_out, dpi=300)
    print(f"Saved dataset plot to: {ds_out}")
    plt.close()

