import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

def plot_truncated(file_path, output_file, dataset_name):
    # Load dataset
    df = pd.read_csv(file_path).dropna()
    
    # 1. Truncate dataset for CV <= 0.25
    df_trunc = df[df['cv'] <= 0.25]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Recall vs CV (Truncated <= 0.25) ---
    s_cv, _ = spearmanr(df_trunc['cv'], df_trunc['recall'])
    p_cv, _ = pearsonr(df_trunc['cv'], df_trunc['recall'])
    
    sns.scatterplot(x=df_trunc['cv'], y=df_trunc['recall'], alpha=0.1, color='#1f77b4', edgecolor=None, s=35, ax=axes[0])
    # Add a regression line to highlight linearity
    sns.regplot(x=df_trunc['cv'], y=df_trunc['recall'], scatter=False, color='red', line_kws={"linewidth": 2, "linestyle": "--"}, ax=axes[0])
    
    axes[0].set_xlim(df_trunc['cv'].min() * 0.95, 0.255)
    axes[0].set_ylim(max(0, df_trunc['recall'].min() - 0.05), 1.05)
    axes[0].set_title(f'Recall vs CV (Truncated ≤ 0.25)\nPearson: {p_cv:.4f} | Spearman: {s_cv:.4f}', fontsize=15, fontweight='bold', pad=12)
    axes[0].set_xlabel('CV (Complexity Variance)', fontsize=14)
    axes[0].set_ylabel('Recall (at ef=100)', fontsize=14)

    # --- Plot 2: Recall vs LID (Full Dataset) ---
    # We use the full dataset for LID to provide a fair baseline comparison
    s_lid, _ = spearmanr(df['lid'], df['recall'])
    p_lid, _ = pearsonr(df['lid'], df['recall'])
    
    lid_max = df['lid'].quantile(0.99)
    sns.scatterplot(x=df['lid'], y=df['recall'], alpha=0.1, color='#2ca02c', edgecolor=None, s=35, ax=axes[1])
    sns.regplot(x=df['lid'], y=df['recall'], scatter=False, color='red', line_kws={"linewidth": 2, "linestyle": "--"}, ax=axes[1])
    
    axes[1].set_xlim(df['lid'].min() * 0.95, lid_max)
    axes[1].set_ylim(max(0, df['recall'].min() - 0.05), 1.05)
    axes[1].set_title(f'Recall vs LID (Full Range)\nPearson: {p_lid:.4f} | Spearman: {s_lid:.4f}', fontsize=15, fontweight='bold', pad=12)
    axes[1].set_xlabel('LID (Local Intrinsic Dimensionality)', fontsize=14)
    axes[1].set_ylabel('Recall (at ef=100)', fontsize=14)

    plt.suptitle(f'{dataset_name} Dataset - Comparing Truncated CV vs LID', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[{dataset_name}] Graph successfully saved to: {output_file}")

plot_truncated('build/metrics_dump_deep.csv', 'deepimage_cv_trunc_0.25_vs_lid.png', 'DeepImage')
plot_truncated('build/metrics_dump.csv', 'glove_cv_trunc_0.25_vs_lid.png', 'GloVe')
