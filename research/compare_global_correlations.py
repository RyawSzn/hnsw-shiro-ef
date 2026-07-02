import pandas as pd
from scipy.stats import spearmanr

def compare(dataset_name, csv_file):
    try:
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['score', 'cv', 'recall'])
        
        # RV (Score) vs Recall
        rv_corr, rv_pval = spearmanr(df['score'], df['recall'])
        
        # CV vs Recall
        cv_corr, cv_pval = spearmanr(df['cv'], df['recall'])
        
        print(f"[{dataset_name}]")
        print(f"  RV vs Recall: {rv_corr:.4f}")
        print(f"  CV vs Recall: {cv_corr:.4f}")
        
        if abs(rv_corr) > abs(cv_corr):
            print(f"  -> RV is STRONGER by {abs(rv_corr) - abs(cv_corr):.4f}")
        else:
            print(f"  -> CV is STRONGER by {abs(cv_corr) - abs(rv_corr):.4f}")
        print("-" * 40)
    except Exception as e:
        print(f"Error on {dataset_name}: {e}")

print("GLOBAL CORRELATION COMPARISON\n" + "="*40)
compare("SIFT (sift-128-euclidean)", "research/sift_score_recall.csv")
compare("GloVe (glove-100-angular)", "research/glove_score_recall.csv")
compare("Deep-Image (deep-image-96-angular)", "research/deep_score_recall.csv")
