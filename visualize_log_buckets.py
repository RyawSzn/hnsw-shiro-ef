#!/usr/bin/env python3
"""visualize_log_buckets.py

Parses gamma_sweep.log to extract the "Initial average recall with ef=100" 
for each of the 10 training buckets across different gamma values.
Plots this initial recall vs bucket index (0-9) to visualize how well 
different gammas separate hard queries from easy queries.
"""

import sys
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from collections import defaultdict

DATASET_LABELS = {
    "deep-image-96-angular": "Deep-Image-96",
    "glove-100-angular":     "GloVe-100",
    "sift-128-euclidean":    "SIFT-128",
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_log_buckets.py gamma_sweep.log", file=sys.stderr)
        sys.exit(1)
        
    log_file = sys.argv[1]
    
    # data[dataset][gamma][bucket_idx] = recall
    data = defaultdict(lambda: defaultdict(dict))
    
    curr_ds = None
    curr_gamma = None
    curr_bucket = None
    
    dataset_re = re.compile(r"Dataset: (\S+)")
    gamma_re = re.compile(r"--- gamma=([0-9.]+) ---")
    bucket_re = re.compile(r"Training rv-bucket (\d+) \[")
    recall_re = re.compile(r"Initial average recall with ef=100: ([0-9.]+)")
    
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            
            m = dataset_re.search(line)
            if m:
                curr_ds = m.group(1)
                continue
                
            m = gamma_re.search(line)
            if m:
                curr_gamma = float(m.group(1))
                continue
                
            m = bucket_re.search(line)
            if m:
                curr_bucket = int(m.group(1))
                continue
                
            m = recall_re.search(line)
            if m and curr_ds and curr_gamma is not None and curr_bucket is not None:
                recall = float(m.group(1))
                data[curr_ds][curr_gamma][curr_bucket] = recall
                # Reset bucket to avoid falsely assigning the next 'ef=150' line
                curr_bucket = None
                
    if not data:
        print("No bucket data found in log.")
        sys.exit(1)
        
    datasets = sorted(data.keys())
    n = len(datasets)
    
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.suptitle("Bucket Separation Power by Gamma\n(Initial Recall at fixed EF=100 for Buckets 0-9)", fontsize=15, y=1.05)
    
    default_gamma = 16.0
    
    for col, ds in enumerate(datasets):
        ax = axes[0, col]
        gammas = sorted(data[ds].keys())
        
        colors = matplotlib.colormaps['viridis'](np.linspace(0, 1, len(gammas)))
        
        for gamma, color in zip(gammas, colors):
            buckets = sorted(data[ds][gamma].keys())
            recalls = [data[ds][gamma][b] for b in buckets]
            
            if gamma == default_gamma:
                lw = 3
                ls = '-'
                alpha = 1.0
                c = 'red'
                label = f"γ={gamma} (default)"
                zorder = 10
            else:
                lw = 1.5
                ls = '-'
                alpha = 0.6
                c = color
                label = f"γ={gamma}"
                zorder = 1
                
            ax.plot(buckets, recalls, marker='o', markersize=4,
                    linewidth=lw, linestyle=ls, color=c, alpha=alpha, label=label, zorder=zorder)
            
        label = DATASET_LABELS.get(ds, ds)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel("Bucket Index (0 = Hardest, 9 = Easiest)", fontsize=11)
        ax.set_ylabel("Initial Recall (ef=100)", fontsize=11)
        
        ax.set_xticks(range(10))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc='best', ncol=2)
        
    plt.tight_layout()
    out_path = Path(__file__).parent / "gamma_log_buckets_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")

if __name__ == "__main__":
    main()
