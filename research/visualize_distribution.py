import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def get_best_ef(df_base, df_res, dataset_name):
    summary_path = "/home/ryawszn/dev/cpp/hnsw-shiro-ef/research/csv/summary_metrics.csv"
    if os.path.exists(summary_path):
        sum_df = pd.read_csv(summary_path)
        ds_sum = sum_df[sum_df["dataset"] == dataset_name]
        adaptive_data = ds_sum[ds_sum["method"] == "adaptive"]
        
        if not adaptive_data.empty:
            adapt_rec = adaptive_data["avg_recall"].values[0]
            base_df = ds_sum[ds_sum["method"] == "baseline"].copy()
            base_df["ef"] = pd.to_numeric(base_df["ef"])
            base_df["recall_diff"] = (base_df["avg_recall"] - adapt_rec).abs()
            return int(base_df.loc[base_df["recall_diff"].idxmin()]["ef"])
            
    adapt_rec = df_res["Recall"].mean()
    unique_efs = df_base["EF"].unique()
    ef_recalls = {ef: df_base[df_base["EF"] == ef]["Recall"].mean() for ef in unique_efs}
    return int(min(ef_recalls, key=lambda ef: abs(ef_recalls[ef] - adapt_rec)))

def main():
    dataset_name = 'msmarco'
    baseline_file = f'research/csv/per_query_baseline_{dataset_name}.csv'
    results_file = f'research/csv/per_query_results_{dataset_name}.csv'
    
    if not os.path.exists(baseline_file) or not os.path.exists(results_file):
        print("CSV files not found. Please ensure you are running this from the project root.")
        return

    df_base = pd.read_csv(baseline_file)
    df_res = pd.read_csv(results_file)

    best_ef = get_best_ef(df_base, df_res, dataset_name)
    df_base_best = df_base[df_base["EF"] == best_ef].copy()

    df_base_best['Type'] = f'Baseline (EF={best_ef})'
    df_res['Type'] = 'Dynamic EF'

    df_all = pd.concat([df_base_best, df_res], ignore_index=True)
    df_all['Latency(ms)'] = df_all['Latency(ns)'] / 1e6

    df_merged = pd.merge(df_base_best, df_res, on="QueryID", suffixes=("_Base", "_Mine"))
    df_merged.sort_values(["Recall_Base", "Latency(ns)_Base"], ascending=[True, False], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    x_percentile = np.linspace(0, 100, len(df_merged))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Query Performance Distribution: Baseline (EF={best_ef}) vs Dynamic EF', fontsize=16)

    sns.boxplot(data=df_all, x='Type', y='Latency(ms)', ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title('Latency Distribution (ms)')
    axes[0, 0].set_ylabel('Latency (ms)')

    sns.boxplot(data=df_all, x='Type', y='Recall', ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title('Recall Distribution')
    axes[0, 1].set_ylabel('Recall')

    sns.scatterplot(data=df_all, x='Latency(ms)', y='Recall', hue='Type', alpha=0.5, ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title('Recall vs Latency (Per Query)')
    axes[1, 0].set_xlabel('Latency (ms)')
    axes[1, 0].set_ylabel('Recall')

    axes[1, 1].plot(x_percentile, df_merged["Recall_Base"], label=f"Baseline Recall (EF={best_ef})", color='#3498db', linewidth=2)
    axes[1, 1].plot(x_percentile, df_merged["Recall_Mine"], label="Dynamic EF Recall", color='#e74c3c', linewidth=2, alpha=0.7)
    axes[1, 1].set_title('Recall vs Ordered Statistic (Percentile)')
    axes[1, 1].set_xlabel('Query Percentile (Sorted by Baseline Recall)')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_xlim(0, 100)

    ax2 = axes[1, 1].twinx()
    ax2.plot(x_percentile, df_merged["EF_Mine"], label="Dynamic EF Value", color='green', linewidth=1, linestyle='--', alpha=0.5)
    ax2.set_ylabel('Dynamic EF Allocation')

    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    output_path = 'research/distribution_visualization.png'
    plt.savefig(output_path, dpi=300)
    print(f"Visualization successfully saved to {output_path}")

if __name__ == "__main__":
    main()

