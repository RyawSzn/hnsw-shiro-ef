import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import warnings

def main():
    csv_path = "research/raw_E_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    e_cols = [c for c in df.columns if c.startswith('E')]
    e_cols.sort(key=lambda x: int(x[1:]), reverse=True)
    
    results = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in e_cols:
            # We use rank(method='first', pct=True) to force exactly 10 equal-sized buckets,
            # even if many queries have E_i = 0.0. 
            df['rank'] = df[col].rank(method='first', pct=True)
            df['bucket_idx'] = pd.cut(df['rank'], bins=10, labels=False) + 1 # 1 to 10
            
            for b in range(1, 11):
                subset = df[df['bucket_idx'] == b]
                if len(subset) < 10:
                    continue
                
                r, _ = spearmanr(subset['m'], subset['recall'])
                if np.isnan(r):
                    r = 0.0
                
                results.append({
                    'Layer': col,
                    'Hardness_Decile': b,
                    'Correlation': r
                })

    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=res_df, x='Hardness_Decile', y='Correlation', hue='Layer', 
                 palette='Set1', marker='o', linewidth=3, markersize=8)
    
    plt.title("Moderator Effect of Individual Layer Hardness ($E_i$)\non the Correlation between Micro Difficulty ($m$) and Recall", 
              fontsize=16, pad=15)
    plt.xlabel("Layer Hardness ($E_i$) Decile (1 = Easiest, 10 = Hardest)", fontsize=13)
    plt.ylabel("Spearman Correlation (m vs Recall)\n<- Stronger Signal (More Negative) | Weaker Signal ->", fontsize=13)
    
    # Add a baseline
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    
    # Invert Y axis so that 'Stronger Correlation' goes UP visually
    # plt.gca().invert_yaxis()  # Sometimes helpful, but let's keep it standard with clear labeling
    
    plt.xticks(range(1, 11))
    plt.legend(title="Routing Layer", fontsize=11, title_fontsize=12, loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_path = "research/figures/Ei_moderator_effect.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
