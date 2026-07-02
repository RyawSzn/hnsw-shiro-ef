import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np

def main():
    print("Loading Glove data (ef=50)...")
    df = pd.read_csv('glove_score_recall.csv')
    df = df.dropna(subset=['score', 'recall'])
    
    # Global Correlation
    p_global, _ = pearsonr(df['score'], df['recall'])
    s_global, _ = spearmanr(df['score'], df['recall'])
    
    print("\n" + "="*50)
    print("GLOBAL CORRELATION (All 10,000 Queries)")
    print("="*50)
    print(f"Pearson:  {p_global:.4f}")
    print(f"Spearman: {s_global:.4f}\n")
    
    # Bucketly Correlation (Aggregated by Deciles)
    # We use qcut to divide into 10 equally sized buckets (deciles)
    df['RV_Bucket'] = pd.qcut(df['score'], q=10, duplicates='drop')
    
    bucket_stats = df.groupby('RV_Bucket').agg(
        Mean_RV=('score', 'mean'),
        Mean_Recall=('recall', 'mean'),
        Query_Count=('score', 'count')
    ).reset_index()
    
    # Correlation of the Bucket Means
    p_bucket, _ = pearsonr(bucket_stats['Mean_RV'], bucket_stats['Mean_Recall'])
    s_bucket, _ = spearmanr(bucket_stats['Mean_RV'], bucket_stats['Mean_Recall'])
    
    print("="*50)
    print("BUCKETED CORRELATION (Aggregated into 10 Deciles)")
    print("="*50)
    print(bucket_stats[['RV_Bucket', 'Mean_RV', 'Mean_Recall', 'Query_Count']].to_string(index=False))
    print(f"\nBucket-Level Pearson:  {p_bucket:.4f}")
    print(f"Bucket-Level Spearman: {s_bucket:.4f}\n")
    
    # Within-Bucket Correlations
    print("="*50)
    print("WITHIN-BUCKET CORRELATIONS (RV vs Recall inside each bucket)")
    print("="*50)
    for name, group in df.groupby('RV_Bucket'):
        if len(group) > 1 and group['score'].nunique() > 1 and group['recall'].nunique() > 1:
            p_in, _ = pearsonr(group['score'], group['recall'])
            s_in, _ = spearmanr(group['score'], group['recall'])
            print(f"Bucket {name}: Pearson = {p_in:+.4f} | Spearman = {s_in:+.4f}")
        else:
            print(f"Bucket {name}: Insufficient variance for correlation")

if __name__ == "__main__":
    main()
