import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

df = pd.read_csv('metrics_dump_deep.csv').dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_theme(style="whitegrid")

cv_max = df['cv'].quantile(0.99)
lid_max = df['lid'].quantile(0.99)
recall_min = df['recall'].min()

s_cv, _ = spearmanr(df['cv'], df['recall'])
sns.scatterplot(x=df['cv'], y=df['recall'], alpha=0.1, color='#1f77b4', edgecolor=None, s=35, ax=axes[0])
axes[0].set_xlim(df['cv'].min() * 0.95, cv_max)
axes[0].set_ylim(max(0, recall_min - 0.05), 1.05)
axes[0].set_title(f'Recall vs CV\nSpearman: {s_cv:.4f}', fontsize=16, fontweight='bold', pad=12)
axes[0].set_xlabel('CV (Complexity Variance)', fontsize=14)
axes[0].set_ylabel('Recall (at ef=100)', fontsize=14)

s_lid, _ = spearmanr(df['lid'], df['recall'])
sns.scatterplot(x=df['lid'], y=df['recall'], alpha=0.1, color='#2ca02c', edgecolor=None, s=35, ax=axes[1])
axes[1].set_xlim(df['lid'].min() * 0.95, lid_max)
axes[1].set_ylim(max(0, recall_min - 0.05), 1.05)
axes[1].set_title(f'Recall vs LID\nSpearman: {s_lid:.4f}', fontsize=16, fontweight='bold', pad=12)
axes[1].set_xlabel('LID (Local Intrinsic Dimensionality)', fontsize=14)
axes[1].set_ylabel('Recall (at ef=100)', fontsize=14)

plt.tight_layout()
output_file = 'recall_cv_lid_zoomed_deep.png'
plt.savefig(output_file, dpi=150)
print(f"Recall vs CV  -> Spearman: {s_cv:.4f}")
print(f"Recall vs LID -> Spearman: {s_lid:.4f}")
print(f"Graph successfully saved to: {output_file}")
