import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import numpy as np

# Load data
df = pd.read_csv('build/metrics_dump.csv')

# Calculate correlations
metrics = ['lid', 'convergence', 'cv']
target = 'recall'

print("--- Correlation with Recall ---")
for m in metrics:
    p_corr, _ = pearsonr(df[target], df[m])
    s_corr, _ = spearmanr(df[target], df[m])
    print(f"{m.upper()}:")
    print(f"  Pearson:  {p_corr:.4f}")
    print(f"  Spearman: {s_corr:.4f}")

# Generate scatter plots
plt.figure(figsize=(18, 5))

for i, m in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    # Using a 2D histogram / hexbin for better visibility if points are dense
    # but scatter with low alpha works too
    sns.scatterplot(x=df[m], y=df[target], alpha=0.1)
    # add a trendline
    sns.regplot(x=df[m], y=df[target], scatter=False, color='red')
    
    plt.title(f'Recall vs {m.upper()}')
    plt.xlabel(m.upper())
    plt.ylabel('Recall (at ef=150)')

plt.tight_layout()
plt.savefig('correlation_plots.png')
print("Plots saved to correlation_plots.png")

# Also, bucket vs convergence analysis
# Let's see the correlation between CV and convergence (are they orthogonal?)
p_corr_cv_convergence, _ = pearsonr(df['cv'], df['convergence'])
print("\n--- Correlation between CV and convergence ---")
print(f"  Pearson: {p_corr_cv_convergence:.4f}")
print("If CV and convergence have low correlation, they capture different aspects of hardness.")


print("\n--- Summary Statistics ---")
print(df.describe())
