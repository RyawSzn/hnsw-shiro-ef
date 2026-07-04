import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

df = pd.read_csv('build/metrics_dump.csv')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.set_theme(style="whitegrid")

cv_max = df['cv'].quantile(0.99)
lid_max = df['lid'].quantile(0.99)
convergence_min = df['convergence'].quantile(0.01)
recall_min = df['recall'].min()

# 1. Recall vs CV
s_cv, _ = spearmanr(df['cv'], df['recall'])
p_cv, _ = pearsonr(df['cv'], df['recall'])
sns.scatterplot(x=df['cv'], y=df['recall'], alpha=0.03, color='#1f77b4', edgecolor=None, s=35, ax=axes[0])
axes[0].set_xlim(df['cv'].min() * 0.95, cv_max)
axes[0].set_ylim(max(0, recall_min - 0.05), 1.05)
axes[0].set_title(f'Recall vs CV\nSpearman: {s_cv:.4f} | Pearson: {p_cv:.4f}', fontsize=15, fontweight='bold', pad=12)
axes[0].set_xlabel('CV (Complexity Variance)', fontsize=13)
axes[0].set_ylabel('2-hop Recall', fontsize=13)

# 2. Recall vs convergence
s_convergence, _ = spearmanr(df['convergence'], df['recall'])
p_convergence, _ = pearsonr(df['convergence'], df['recall'])
sns.scatterplot(x=df['convergence'], y=df['recall'], alpha=0.03, color='#ff7f0e', edgecolor=None, s=35, ax=axes[1])
axes[1].set_xlim(convergence_min * 0.95, df['convergence'].max() * 1.02)
axes[1].set_ylim(max(0, recall_min - 0.05), 1.05)
axes[1].set_title(f'Recall vs convergence (Convergence)\nSpearman: {s_convergence:.4f} | Pearson: {p_convergence:.4f}', fontsize=15, fontweight='bold', pad=12)
axes[1].set_xlabel('convergence (Revisit Rank / Convergence)', fontsize=13)
axes[1].set_ylabel('2-hop Recall', fontsize=13)

# 3. Recall vs LID
s_lid, _ = spearmanr(df['lid'], df['recall'])
p_lid, _ = pearsonr(df['lid'], df['recall'])
sns.scatterplot(x=df['lid'], y=df['recall'], alpha=0.03, color='#2ca02c', edgecolor=None, s=35, ax=axes[2])
axes[2].set_xlim(df['lid'].min() * 0.95, lid_max)
axes[2].set_ylim(max(0, recall_min - 0.05), 1.05)
axes[2].set_title(f'Recall vs LID\nSpearman: {s_lid:.4f} | Pearson: {p_lid:.4f}', fontsize=15, fontweight='bold', pad=12)
axes[2].set_xlabel('LID (Local Intrinsic Dimensionality)', fontsize=13)
axes[2].set_ylabel('2-hop Recall', fontsize=13)

plt.tight_layout()
output_file = 'recall_all_metrics_2hop.png'
plt.savefig(output_file, dpi=150)
print(f"Graph successfully saved to: {output_file}")
