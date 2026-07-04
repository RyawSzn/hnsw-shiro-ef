import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Load the previously generated metrics for glove-100-angular
df = pd.read_csv('build/metrics_dump.csv')

# Set up the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_theme(style="whitegrid")

# 1. Recall vs CV
p_cv, _ = pearsonr(df['cv'], df['recall'])
s_cv, _ = spearmanr(df['cv'], df['recall'])

sns.scatterplot(x=df['cv'], y=df['recall'], alpha=0.1, color='blue', ax=axes[0])
sns.regplot(x=df['cv'], y=df['recall'], scatter=False, color='red', line_kws={"linewidth": 2}, ax=axes[0])
axes[0].set_title(f'Recall vs CV\nPearson: {p_cv:.4f} | Spearman: {s_cv:.4f}', fontsize=14, pad=10)
axes[0].set_xlabel('CV (Complexity Variance)', fontsize=12)
axes[0].set_ylabel('Recall (at ef=100)', fontsize=12)

# 2. Recall vs LID
p_lid, _ = pearsonr(df['lid'], df['recall'])
s_lid, _ = spearmanr(df['lid'], df['recall'])

sns.scatterplot(x=df['lid'], y=df['recall'], alpha=0.1, color='green', ax=axes[1])
sns.regplot(x=df['lid'], y=df['recall'], scatter=False, color='red', line_kws={"linewidth": 2}, ax=axes[1])
axes[1].set_title(f'Recall vs LID\nPearson: {p_lid:.4f} | Spearman: {s_lid:.4f}', fontsize=14, pad=10)
axes[1].set_xlabel('LID (Local Intrinsic Dimensionality)', fontsize=12)
axes[1].set_ylabel('Recall (at ef=100)', fontsize=12)

# Save the plot
plt.tight_layout()
output_file = 'recall_cv_lid_correlation.png'
plt.savefig(output_file, dpi=150)
print(f"Graph successfully saved to: {output_file}")
print("--- Correlation Data ---")
print(f"Recall vs CV  -> Pearson: {p_cv:.4f}, Spearman: {s_cv:.4f}")
print(f"Recall vs LID -> Pearson: {p_lid:.4f}, Spearman: {s_lid:.4f}")
