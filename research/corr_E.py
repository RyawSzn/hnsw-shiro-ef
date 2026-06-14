import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os
import warnings

def main():
    csv_path = "research/raw_E_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Identify E columns (e.g., E5, E4, E3, E2, E1)
    e_cols = [c for c in df.columns if c.startswith('E')]
    
    # Sort them descending (from top layer E_L down to E_1)
    e_cols.sort(key=lambda x: int(x[1:]), reverse=True)
    
    print("=============================================================")
    print("   Correlation: Individual Layer Hardness (E_i) vs Recall    ")
    print("=============================================================")
    print(f"{'Layer':<8} | {'Pearson (Linear)':<20} | {'Spearman (Rank)'}")
    print("-" * 61)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in e_cols:
            # Handle edge cases where a layer might have exactly zero variance (e.g. top layer E5)
            if df[col].nunique() <= 1:
                r, rho = 0.0, 0.0
            else:
                r, _ = pearsonr(df[col], df['recall'])
                rho, _ = spearmanr(df[col], df['recall'])
            
            print(f"{col:<8} | {r:>10.4f}           | {rho:>10.4f}")
            
    print("=============================================================")
    print("Note: Hardness (E_i) naturally has a NEGATIVE correlation.")
    print("      (Higher hardness score -> Lower final recall)")

if __name__ == "__main__":
    main()
