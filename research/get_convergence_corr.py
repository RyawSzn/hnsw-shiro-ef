import pandas as pd
from scipy.stats import pearsonr, spearmanr
df = pd.read_csv('build/metrics_dump.csv')
p_convergence, _ = pearsonr(df['convergence'], df['recall'])
s_convergence, _ = spearmanr(df['convergence'], df['recall'])
print(f"Recall vs convergence -> Pearson: {p_convergence:.4f}, Spearman: {s_convergence:.4f}")
print("Mean recall at 2-hop:", df['recall'].mean())
