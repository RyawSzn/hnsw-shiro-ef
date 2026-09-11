import os
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.stats import pearsonr, spearmanr
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def compute_lid_from_dists(all_query_distances, k, num_queries, epsilon=1e-1):
    lids = np.zeros(num_queries)
    for i in range(num_queries):
        dists = all_query_distances[i][:k]
        dists = dists[dists > 0]
        if len(dists) == 0:
            lids[i] = np.nan
            continue
        max_dist = np.max(dists)
        if max_dist == 0:
            max_dist = epsilon
        
        log_values = np.log(dists / max_dist)
        with np.errstate(divide='ignore', invalid='ignore'):
            lid = -1 / np.mean(log_values)
            lids[i] = np.nan if np.isnan(lid) or np.isinf(lid) else lid
    return lids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name, e.g., sift-128-euclidean')
    parser.add_argument('--k_lid', type=int, default=100, help='k for LID calculation')
    args = parser.parse_args()

    dataset = args.dataset
    experiments_root = os.environ.get("EXPERIMENTS_ROOT")
    if not experiments_root:
        print("Error: EXPERIMENTS_ROOT environment variable is not set.")
        sys.exit(1)

    hdf5_path = os.path.join(experiments_root, "data", f"{dataset}.hdf5")
    
    cv_dir = "research/csv"
    os.makedirs(cv_dir, exist_ok=True)
    cv_recall_file = f"{cv_dir}/cv_recall_{dataset}.csv"

    if not os.path.exists(cv_recall_file):
        print(f"Error: {cv_recall_file} does not exist. Please run extract_cv_recall C++ binary first.")
        sys.exit(1)

    print(f"Loading CV and Recall from {cv_recall_file}...")
    df = pd.read_csv(cv_recall_file)

    print(f"Loading HDF5 data from {hdf5_path} to compute LID...")
    with h5py.File(hdf5_path, 'r') as f:
        distances = np.array(f['distances'])
    
    num_queries = distances.shape[0]
    print(f"Computing LID for {num_queries} queries (k={args.k_lid})...")
    lids = compute_lid_from_dists(distances, args.k_lid, num_queries)
    
    df['lid'] = lids

    combined_csv = f"{cv_dir}/metrics_dump_{dataset}.csv"
    df.to_csv(combined_csv, index=False)
    print(f"Saved combined metrics to {combined_csv}")

    # Drop NaNs before correlation
    df_clean = df.dropna()

    print("\n--- Correlations (Full Range) ---")
    
    # CV vs Recall
    p_cv, _ = pearsonr(df_clean['cv'], df_clean['recall'])
    s_cv, _ = spearmanr(df_clean['cv'], df_clean['recall'])
    print(f"Recall vs CV  -> Pearson: {p_cv:.4f} | Spearman: {s_cv:.4f}")

    # LID vs Recall
    p_lid, _ = pearsonr(df_clean['lid'], df_clean['recall'])
    s_lid, _ = spearmanr(df_clean['lid'], df_clean['recall'])
    print(f"Recall vs LID -> Pearson: {p_lid:.4f} | Spearman: {s_lid:.4f}")

    print("\n--- Correlations (Capped CV <= 0.25) ---")
    df_capped = df_clean.copy()
    df_capped.loc[df_capped['cv'] > 0.25, 'cv'] = 0.25
    
    if not df_capped.empty:
        if len(df_capped) > 1:
            p_cv_capped, _ = pearsonr(df_capped['cv'], df_capped['recall'])
            s_cv_capped, _ = spearmanr(df_capped['cv'], df_capped['recall'])
            print(f"Recall vs CV (Capped at 0.25) -> Pearson: {p_cv_capped:.4f} | Spearman: {s_cv_capped:.4f}")
        else:
            print("Not enough data points to compute correlation.")
            p_cv_capped, s_cv_capped = 0.0, 0.0
    else:
        print("No data points found.")
        p_cv_capped, s_cv_capped = 0.0, 0.0

    print("\n=== Generating PNG Visualization ===")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Recall vs CV (Capped <= 0.25)
    if not df_capped.empty and len(df_capped) > 1:
        sns.scatterplot(x=df_capped['cv'], y=df_capped['recall'], alpha=0.1, color='#1f77b4', edgecolor=None, s=35, ax=axes[0])
        sns.regplot(x=df_capped['cv'], y=df_capped['recall'], scatter=False, color='red', line_kws={"linewidth": 2, "linestyle": "--"}, ax=axes[0])
        
        cv_min = df_capped['cv'].min() * 0.95
        cv_max = 0.255
        axes[0].set_xlim(cv_min, cv_max)
        axes[0].set_ylim(max(0, df_capped['recall'].min() - 0.05), 1.05)
        axes[0].set_title(f'Recall vs CV (Capped at 0.25)\nPearson: {p_cv_capped:.4f} | Spearman: {s_cv_capped:.4f}', fontsize=15, fontweight='bold', pad=12)
    else:
        axes[0].text(0.5, 0.5, "Not enough data points", ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("Recall vs CV (Capped at 0.25)")
        
    axes[0].set_xlabel('CV (Complexity Variance)', fontsize=14)
    axes[0].set_ylabel('Recall (at ef=100)', fontsize=14)

    # Plot 2: Recall vs LID (Full Dataset)
    lid_max = df_clean['lid'].quantile(0.99)
    sns.scatterplot(x=df_clean['lid'], y=df_clean['recall'], alpha=0.1, color='#2ca02c', edgecolor=None, s=35, ax=axes[1])
    sns.regplot(x=df_clean['lid'], y=df_clean['recall'], scatter=False, color='red', line_kws={"linewidth": 2, "linestyle": "--"}, ax=axes[1])
    
    axes[1].set_xlim(df_clean['lid'].min() * 0.95, lid_max)
    axes[1].set_ylim(max(0, df_clean['recall'].min() - 0.05), 1.05)
    axes[1].set_title(f'Recall vs LID (Full Range)\nPearson: {p_lid:.4f} | Spearman: {s_lid:.4f}', fontsize=15, fontweight='bold', pad=12)
    axes[1].set_xlabel('LID (Local Intrinsic Dimensionality)', fontsize=14)
    axes[1].set_ylabel('Recall (at ef=100)', fontsize=14)

    plt.suptitle(f'{dataset} Dataset - Comparing Capped CV vs LID', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    img_dir = "research/img/corr"
    os.makedirs(img_dir, exist_ok=True)
    png_path = f"{img_dir}/{dataset}_cv_trunc_0.25_vs_lid.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Graph successfully saved to: {png_path}")

if __name__ == "__main__":
    main()
