import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import os

def main():
    csv_path = "research/compare_M_metrics.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # We bucket M_Graph and M_RC into 10 deciles (1 = Easiest, 10 = Hardest)
    df['M_Graph_Decile'] = pd.qcut(df['M_Graph'], q=10, labels=range(1, 11), duplicates='drop')
    df['M_RC_Decile'] = pd.qcut(df['M_RC'], q=10, labels=range(1, 11), duplicates='drop')

    print("==========================================================================================")
    print("      Testing the Moderator Effect: Graph Traversal (M_Graph) vs Relative Contrast (M_RC) ")
    print("      Hypothesis: A strong moderator makes the (m -> recall) correlation plummet in Hard M")
    print("==========================================================================================")
    print(f"{'Decile':<8} | {'Avg Recall (Graph)':<20} | {'Spearman (m vs Recall) for M_Graph'}")
    print("-" * 85)

    graph_res = []
    for b in range(1, 11):
        subset = df[df['M_Graph_Decile'] == b]
        r, _ = spearmanr(subset['m_LID'], subset['recall'])
        avg_r = subset['recall'].mean()
        graph_res.append({'Decile': b, 'Correlation': r, 'Metric': 'M_Graph (Traversal)'})
        print(f"M{b:<7} | {avg_r:<20.4f} | {r:>10.4f}")

    print("\n------------------------------------------------------------------------------------------")
    print(f"{'Decile':<8} | {'Avg Recall (RC)':<20} | {'Spearman (m vs Recall) for M_RC'}")
    print("-" * 85)

    rc_res = []
    for b in range(1, 11):
        subset = df[df['M_RC_Decile'] == b]
        r, _ = spearmanr(subset['m_LID'], subset['recall'])
        avg_r = subset['recall'].mean()
        rc_res.append({'Decile': b, 'Correlation': r, 'Metric': 'M_RC (Relative Contrast)'})
        print(f"M{b:<7} | {avg_r:<20.4f} | {r:>10.4f}")

    print("==========================================================================================\n")

    res_df = pd.DataFrame(graph_res + rc_res)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x='Decile', y='Correlation', hue='Metric', 
                 palette=['#1f77b4', '#ff7f0e'], marker='o', linewidth=3, markersize=9)

    plt.title("Moderator Effect Comparison: Graph Traversal vs Relative Contrast\n$\\widehat{M}(q) = \widehat{d}_{mean}(q) / b_0(q)$ vs $\\sum w_l E_l$", fontsize=15, pad=15)
    plt.xlabel("M Decile (1 = Easiest Routing, 10 = Hardest Routing)", fontsize=12)
    plt.ylabel("Spearman Correlation (m vs Recall)", fontsize=12)
    
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_path = "research/figures/compare_M_moderators.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization plot to {out_path}")


if __name__ == "__main__":
    main()
