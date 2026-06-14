import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import os

def main():
    csv_path = "research/compare_metrics.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    metrics = [
        ('LID (Continuous)', 'm_LID'),
        ('Eps-Hardness (eps=0.1)', 'e_eps01'),
        ('Eps-Hardness (eps=0.2)', 'e_eps02'),
        ('Eps-Hardness (eps=0.5)', 'e_eps05')
    ]

    print("==========================================================================")
    print("      Comparing Micro Difficulty Metrics: LID vs Probe-Local Eps-Hardness ")
    print("==========================================================================")
    print(f"{'Metric':<25} | {'Unique Values':<15} | {'Spearman Correlation with Recall'}")
    print("-" * 74)

    for name, col in metrics:
        unique_vals = df[col].nunique()
        
        # Calculate correlation with recall
        # Higher LID -> lower recall, so corr is negative
        # Higher Eps-Hardness (more points close to nearest neighbor) -> denser -> harder -> lower recall
        r, _ = spearmanr(df[col], df['recall'])
        
        print(f"{name:<25} | {unique_vals:<15} | {r:>10.4f}")

    print("==========================================================================\n")
    
    # Let's show the problem with percentile bucketing for Eps-Hardness
    print("Example of the 'Ties' Problem when calculating Percentile Buckets:")
    print("Top 5 most common values for Eps-Hardness (eps=0.1):")
    counts = df['e_eps01'].value_counts().head(5)
    total = len(df)
    for val, count in counts.items():
        print(f"  Value: {val} -> Count: {count} queries ({(count/total)*100:.1f}% of dataset)")
        
    print("\nCompare to Top 5 most common values for LID:")
    counts = df['m_LID'].value_counts().head(5)
    for val, count in counts.items():
        print(f"  Value: {val:.4f} -> Count: {count} queries ({(count/total)*100:.1f}% of dataset)")


if __name__ == "__main__":
    main()
