import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    csv_path = "research/lookup_table_RC_10x10.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Pivot the data into 10x10 matrices
    pivot_ef = df.pivot(index='M_bin', columns='m_bin', values='ef')
    pivot_recall = df.pivot(index='M_bin', columns='m_bin', values='actual_recall')

    # Create a figure with 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # --- Plot 1: The Target EF Heatmap ---
    sns.heatmap(pivot_ef, ax=axes[0], annot=True, fmt=".0f", cmap="YlOrRd", 
                cbar_kws={'label': 'Adaptive ef Required'}, annot_kws={"size": 11})
    
    axes[0].set_title("Optimal 'ef' to Reach 95% Recall (10x10 Table)", fontsize=15)
    axes[0].set_xlabel("Local Ambiguity ($m_{LID}$) Decile -> Harder", fontsize=12)
    axes[0].set_ylabel("Relative Contrast ($M_{RC}$) Decile -> Harder", fontsize=12)
    
    axes[0].set_yticklabels([f"M{int(float(y.get_text()))}" for y in axes[0].get_yticklabels()], rotation=0)
    axes[0].set_xticklabels([f"m{int(float(x.get_text()))}" for x in axes[0].get_xticklabels()])

    # --- Plot 2: The Actual Recall Heatmap ---
    sns.heatmap(pivot_recall, ax=axes[1], annot=True, fmt=".3f", cmap="crest",
                vmin=0.90, vmax=1.00, cbar_kws={'label': 'Actual Recall Achieved'}, annot_kws={"size": 11})
    
    axes[1].set_title("Actual Recall Achieved (Target was 0.95)", fontsize=15)
    axes[1].set_xlabel("Local Ambiguity ($m_{LID}$) Decile -> Harder", fontsize=12)
    axes[1].set_ylabel("") 
    
    axes[1].set_yticklabels([f"M{int(float(y.get_text()))}" for y in axes[1].get_yticklabels()], rotation=0)
    axes[1].set_xticklabels([f"m{int(float(x.get_text()))}" for x in axes[1].get_xticklabels()])

    plt.tight_layout()
    out_path = "research/figures/lookup_ef_heatmap_RC_10x10.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    
    print(f"Saved Visualization to: {out_path}")

if __name__ == "__main__":
    main()
