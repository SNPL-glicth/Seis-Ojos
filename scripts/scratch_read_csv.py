import os
import pandas as pd
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "tsb_ad", "TSB-AD", "benchmark_exp", "benchmark_eval_results", "uni_mergedTable_VUS-PR.csv"))
df = pd.read_csv(csv_path)
detectors = df.columns[1:33]
means = df[detectors].mean().sort_values(ascending=False)
for k, v in means.items():
    print(f"{k}: {v:.4f}")
