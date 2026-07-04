import pandas as pd
import numpy as np

df = pd.read_csv('build/metrics_dump.csv')
print("CV ranges from", df['cv'].min(), "to", df['cv'].max())
print("Most CV concentrated in", df['cv'].quantile(0.1), "to", df['cv'].quantile(0.9))
