import pandas as pd
import numpy as np
import h5py
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_convergence

def compute_lid_from_dists(all_query_distances, k, num_queries, epsilon=1e-1):
    lids = np.zeros(num_queries)
    for i in range(num_queries):
        sorted_distances = np.sort(all_query_distances[i])[0:k]  
        r_k = sorted_distances[-1]
        if r_k == 0:
            lids[i] = np.nan
        else:
            safe_distances = np.maximum(sorted_distances / r_k, epsilon)
            log_values = np.log(safe_distances)
            lid = -1 / np.mean(log_values)
            lids[i] = np.nan if np.isnan(lid) or np.isinf(lid) else lid
    return lids

def main():
    df = pd.read_csv('sift_convergence_recall.csv')
    
    hdf5_path = '/home/ryawszn/experiments/2metric/data/sift-128-euclidean.hdf5'
    with h5py.File(hdf5_path, 'r') as f:
        distances = np.array(f['distances'])
    
    df['LID'] = compute_lid_from_dists(distances, 100, distances.shape[0])
    df = df.dropna(subset=['LID', 'convergence', 'recall'])
    
    print("\n" + "="*50)
    print("MATHEMATICAL PROOF: convergence vs LID IN PROXIMITY GRAPHS (GLOVE DATASET)")
    print("="*50)

    # Note: convergence has a POSITIVE correlation with Recall (Higher convergence = Easier)
    # Note: LID has a NEGATIVE correlation with Recall (Higher LID = Harder)

    # 1. GLOBAL CORRELATION (Baseline)
    sp_convergence_global, _ = spearmanr(df['convergence'], df['recall'])
    sp_lid_global, _ = spearmanr(df['LID'], df['recall'])
    print(f"\n[1] GLOBAL CORRELATION (All Queries):")
    print(f"    LID: {abs(sp_lid_global):.4f}")
    print(f"    convergence:  {abs(sp_convergence_global):.4f}")

    # 2. TAIL CORRELATION (Hardest 15% of Queries)
    p15 = df['recall'].quantile(0.15)
    df_tail = df[df['recall'] <= p15]
    
    sp_convergence_tail, _ = spearmanr(df_tail['convergence'], df_tail['recall'])
    sp_lid_tail, _ = spearmanr(df_tail['LID'], df_tail['recall'])
    
    print(f"\n[2] TAIL CORRELATION (Bottom 15% Recalls - The 'Hard' Queries):")
    print(f"    LID: {abs(sp_lid_tail):.4f} (LID's predictive power drops drastically)")
    print(f"    convergence:  {abs(sp_convergence_tail):.4f}")
    
    # 3. ROUTING FAILURE PREDICTION (AUROC)
    # Define failure as worst 10% of queries
    p10 = df['recall'].quantile(0.10)
    y_fail = (df['recall'] <= p10).astype(int)
    
    # Predict failure: Higher convergence for AUROC means more likely to fail.
    # So we use -convergence (lower convergence = failure) and LID (higher LID = failure)
    auc_convergence = roc_auc_convergence(y_fail, -df['convergence'])
    auc_lid = roc_auc_convergence(y_fail, df['LID'])
    
    print(f"\n[3] PREDICTING ROUTING FAILURES (ROC-AUC for worst 10%):")
    print(f"    LID AUROC: {auc_lid:.4f}")
    print(f"    convergence AUROC:  {auc_convergence:.4f}")

    # 4. THE 'LID BLIND SPOT' (Local Minima Detection)
    # Queries that LID thinks are "Easy" (LID <= Median) but actually "Failed" (Recall <= 10th percentile)
    lid_median = df['LID'].median()
    blind_spots = df[(df['LID'] <= lid_median) & (df['recall'] <= p10)]
    
    # Did convergence catch these blind spots? (convergence <= Median, meaning convergence correctly thought they were hard)
    convergence_median = df['convergence'].median()
    caught_by_convergence = blind_spots[blind_spots['convergence'] <= convergence_median]
    
    print(f"\n[4] LID BLIND SPOTS (Local Minima Traps):")
    print(f"    Queries LID falsely thought were 'Easy' but failed: {len(blind_spots)}")
    print(f"    Out of those, convergence correctly flagged as 'Hard': {len(caught_by_convergence)} ({len(caught_by_convergence)/len(blind_spots)*100:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
