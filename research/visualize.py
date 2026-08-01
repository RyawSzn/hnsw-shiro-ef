import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import numpy as np

def create_boxplot(df_subset, output_path, title_suffix=""):
    # Filter out the top 1% extreme latency outliers
    q99 = df_subset['Latency(ns)'].quantile(0.99)
    df_filtered = df_subset[df_subset['Latency(ns)'] <= q99].copy()
    
    # Round recall to 2 decimal places to group them cleanly
    df_filtered['Recall_Cat'] = df_filtered['Recall'].round(2).astype(str)
    
    # Clean up the graph: Remove recall levels that have fewer than 3 queries
    # This prevents those weird "single vertical line" boxes at the bottom of the plot
    counts = df_filtered['Recall_Cat'].value_counts()
    valid_cats = counts[counts >= 3].index
    df_filtered = df_filtered[df_filtered['Recall_Cat'].isin(valid_cats)]
    
    # Sort recall categories for the Y-axis (highest recall at the top)
    order = sorted(df_filtered['Recall_Cat'].unique(), key=float, reverse=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal boxplot: Latency on X, Recall on Y
    sns.boxplot(data=df_filtered, x='Latency(ns)', y='Recall_Cat', 
                order=order, ax=ax, color='skyblue', fliersize=2)
                
    # Calculate medians for the red trend line
    medians = [df_filtered[df_filtered['Recall_Cat'] == cat]['Latency(ns)'].median() for cat in order]
    y_coords = np.arange(len(order))
    
    # Plot the red curve connecting the medians
    ax.plot(medians, y_coords, color='red', marker='o', linestyle='-', linewidth=2, zorder=5, label='Median Trend')
    
    ax.set_title(f'Latency Distribution Grouped by Recall Level {title_suffix}', fontsize=14, pad=15)
    ax.set_xlabel('Latency (ns)', fontsize=12)
    ax.set_ylabel('Recall Level', fontsize=12)
    ax.legend()
    
    # Add gridlines on the X axis to easily read latency values
    ax.grid(True, alpha=0.4, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig) # Close the figure to free up memory
    print(f"Visualization saved to: {output_path}")

def visualize(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    base_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # If there is an 'EF' column, split by EF and generate a separate file for each
    if 'EF' in df.columns:
        ef_values = df['EF'].unique()
        print(f"Detected 'EF' column with values: {ef_values}")
        for ef in sorted(ef_values):
            df_ef = df[df['EF'] == ef].copy()
            output_path = os.path.join(base_dir, f"{base_name}_ef{ef}_boxplot_horizontal.png")
            create_boxplot(df_ef, output_path, title_suffix=f"(EF={ef})")
    else:
        # Generate a single plot if no 'EF' column
        output_path = os.path.join(base_dir, f"{base_name}_boxplot_horizontal.png")
        create_boxplot(df, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize per-query results CSV.')
    parser.add_argument('csv_path', type=str, nargs='?', 
                        default='csv/per_query_results_deep-image-96-angular.csv',
                        help='Path to the CSV file')
    args = parser.parse_args()
    
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, args.csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path
            
    visualize(csv_path)
