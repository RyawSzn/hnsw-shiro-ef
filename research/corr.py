import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    csv_path = "research/raw_E_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Sample exactly 5000 queries
    df = df.sample(n=5000, random_state=42).copy()
    
    # Identify available E columns (now E5, E4, E3, E2, E1)
    E_cols = [c for c in df.columns if c.startswith('E')]
    
    # Apply your exact formula: w_l = 2^l / sum(2^j)
    weights = {}
    for col in E_cols:
        l = int(col[1:]) # extract the integer l from 'E_l'
        weights[col] = 2.0 ** l
        
    # Normalize weights by sum(2^j)
    tot_weight = sum(weights.values())
    for k in weights:
        weights[k] /= tot_weight

    print("=== Weight Distribution: w_l = 2^l / sum(2^j) ===")
    for k in sorted(weights.keys(), reverse=True):
        print(f"  {k}: {weights[k]:.4f}")
    print("===============================================\n")
    
    # Compute M
    df['M'] = 0.0
    for col, w in weights.items():
        df['M'] += df[col] * w
            
    # Create 20x20 Uniform Count (Quantile) Buckets
    df['M_Bin'] = pd.qcut(df['M'], q=20, labels=[f"M{i}" for i in range(1, 21)], duplicates='drop')
    df['m_Bin'] = pd.qcut(df['m'], q=20, labels=[f"m{i}" for i in range(1, 21)], duplicates='drop')
    
    pivot = pd.pivot_table(df, values='recall', index='M_Bin', columns='m_Bin', aggfunc='mean')
    
    print("===================================================================================================================================")
    print("                      20x20 Recall Matrix (5k Queries) using w_l = 2^l / sum(2^j)                                                  ")
    print("===================================================================================================================================")
    
    header = "    |" + "".join([f"{col:>5}" for col in pivot.columns])
    print(header)
    print("-" * len(header))
    
    for index, row in pivot.iterrows():
        row_str = f"{index:<3} |"
        for val in row:
            if pd.isna(val):
                row_str += "  -  " 
            else:
                row_str += f"{val:5.2f}"
        print(row_str)
    print("===================================================================================================================================")
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm_r", cbar_kws={'label': 'Average Recall'}, annot_kws={"size": 7})
    plt.title(f"20x20 Average Recall Matrix (5,000 Queries)\nM Computed strictly with w_l = 2^l / sum(2^j)")
    plt.xlabel("Micro Difficulty (m) Quantiles -> Harder")
    plt.ylabel("Macro Difficulty (M) Quantiles -> Harder")
    
    out_path = "research/figures/recall_20x20_heatmap_formula.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved Heatmap plot to {out_path}")

if __name__ == "__main__":
    main()
