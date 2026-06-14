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

    # We currently have M_bin from 1 to 20. Let's group them into 5 Regimes.
    # Regime 1: Bins 1-4
    # Regime 2: Bins 5-8
    # Regime 3: Bins 9-12
    # Regime 4: Bins 13-16
    # Regime 5: Bins 17-20
    df['M_Regime'] = pd.cut(df['M_bin'], bins=[0, 4, 8, 12, 16, 20], 
                            labels=['Very Easy Routing (M)', 'Easy Routing (M)', 
                                    'Moderate Routing (M)', 'Hard Routing (M)', 'Very Hard Routing (M)'])
                                    
    # We group m_bin from 1 to 20 into 10 buckets (pairs)
    df['m_Decile'] = pd.cut(df['m_bin'], bins=10, labels=range(1, 11))
    
    # Calculate the average required 'ef'
    agg_df = df.groupby(['m_Decile', 'M_Regime'], observed=True)['ef'].mean().reset_index()

    plt.figure(figsize=(11, 7))
    
    sns.lineplot(data=agg_df, x='m_Decile', y='ef', hue='M_Regime', 
                 palette='rocket_r', linewidth=4, marker='o', markersize=10)

    plt.title("Routing Difficulty (M) Acts as a Moderator for Local Density (m)\n$\\widehat{ef}(q) = g_{M\\text{-regime}}(m(q))$", fontsize=16, pad=15)
    plt.xlabel("Micro Difficulty ($m$) Deciles -> Exponentially Harder", fontsize=13)
    plt.ylabel("Required 'ef' Budget to Rescue 95% Recall", fontsize=13)
    
    plt.legend(title="Macro Routing Regime ($M$)", fontsize=11, title_fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    out_path = "research/figures/moderator_effect_5_regimes.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved Moderator plot to {out_path}")

    # Print the 5x10 matrix
    pivot = agg_df.pivot(index='M_Regime', columns='m_Decile', values='ef')
    print("\n=========================================================================")
    print("                     5x10 Lookup Table (M Regime x m Decile)             ")
    print("=========================================================================")
    header = "          |" + "".join([f"   m{col:<3}" for col in pivot.columns])
    print(header)
    print("-" * len(header))
    for index, row in pivot.iterrows():
        row_str = f"{index[:10]:<9} |"
        for val in row:
            if pd.isna(val):
                row_str += "  ---  "
            else:
                row_str += f"{val:6.0f}"
        print(row_str)
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
