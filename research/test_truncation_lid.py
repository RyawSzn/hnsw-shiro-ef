import pandas as pd
from scipy.stats import spearmanr, pearsonr

def print_corrs(file_path, dataset_name):
    df = pd.read_csv(file_path).dropna()
    df_trunc = df[df['cv'] <= 0.25]
    
    cv_s, _ = spearmanr(df_trunc['cv'], df_trunc['recall'])
    cv_p, _ = pearsonr(df_trunc['cv'], df_trunc['recall'])
    
    lid_s, _ = spearmanr(df['lid'], df['recall'])
    lid_p, _ = pearsonr(df['lid'], df['recall'])
    
    print(f"[{dataset_name}] Truncated CV (<=0.25): Spearman={cv_s:.4f}, Pearson={cv_p:.4f}")
    print(f"[{dataset_name}] Full LID: Spearman={lid_s:.4f}, Pearson={lid_p:.4f}")

print_corrs('build/metrics_dump.csv', 'GloVe')
print_corrs('build/metrics_dump_deep.csv', 'DeepImage')
