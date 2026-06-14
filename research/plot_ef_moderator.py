import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    csv_path = "research/lookup_table.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Group M into 3 theoretical regimes
    def categorize_M(m_bin):
        if m_bin <= 7:
            return "Low M (Easy Routing)"
        elif m_bin <= 14:
            return "Mid M (Moderate Routing)"
        else:
            return "High M (Hard Routing)"

    df['M_Regime'] = df['M_bin'].apply(categorize_M)
    
    # Calculate the average 'ef' required for each (m_bin, M_Regime) combination
    agg_df = df.groupby(['m_bin', 'M_Regime'])['ef'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=agg_df, x='m_bin', y='ef', hue='M_Regime', 
                 palette=['#2ca02c', '#f1c40f', '#d62728'], 
                 linewidth=3, marker='o', markersize=8)

    plt.title("The Moderator Effect of M on Required 'ef'\n$\\widehat{ef}(q) = g_{M\\text{-regime}}(m(q))$", fontsize=16, pad=15)
    plt.xlabel("Micro Difficulty (m) Bin -> Harder Neighborhood", fontsize=12)
    plt.ylabel("Required 'ef' to Hit 95% Recall", fontsize=12)
    
    # Customize the legend
    plt.legend(title="Routing Regime (M)", fontsize=11, title_fontsize=12, loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    out_path = "research/figures/ef_moderator_effect.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved Moderator Effect plot to {out_path}")

if __name__ == "__main__":
    main()
