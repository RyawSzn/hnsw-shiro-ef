import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def evaluate(dataset_name, csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['score', 'cv', 'recall'])
    
    # Sort by cv
    df = df.sort_values(by='cv').reset_index(drop=True)
    
    n_buckets = 10
    bucket_size = len(df) // n_buckets
    
    print(f"Dataset: {dataset_name}")
    print(f"{'Bucket':<8} | {'CV Range':<35} | {'Size':<6} | {'Spearman Correlation (RV vs Recall)':<25}")
    print("-" * 85)
    
    correlations = []
    
    for i in range(n_buckets):
        start_idx = i * bucket_size
        # For the last bucket, include any remaining elements
        end_idx = (i + 1) * bucket_size if i < n_buckets - 1 else len(df)
        
        bucket_df = df.iloc[start_idx:end_idx]
        
        cv_min = bucket_df['cv'].min()
        cv_max = bucket_df['cv'].max()
        
        sp_corr, sp_pval = spearmanr(bucket_df['score'], bucket_df['recall'])
        correlations.append(sp_corr)
        
        cv_range_str = f"[{cv_min:.6f}, {cv_max:.6f}]"
        print(f"{i:<8} | {cv_range_str:<35} | {len(bucket_df):<6} | {sp_corr:.4f} (p={sp_pval:.4f})")
    
    print("-" * 85)
    print(f"Average Correlation across buckets: {np.mean(correlations):.4f}\n")

evaluate("Deep-Image", "research/deep_score_recall.csv")
