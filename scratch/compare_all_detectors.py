import pandas as pd
from pathlib import Path

def get_row(name):
    p = Path("../benchmarks/nab/results") / name / f"{name}_standard_scores.csv"
    df = pd.read_csv(p)
    return df[df["Detector"] == "Totals"].iloc[0]

detectors = [
    "null",
    "random",
    "expose",
    "skyline",
    "windowedGaussian",
    "sixeyes_seasonal_pct",
    "sixeyes_peak_decay",
    "twitterADVec",
    "relativeEntropy",
    "randomCutForest",
    "earthgeckoSkyline",
]

print(f"{'Detector':<22} | {'Raw Score':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<7}")
print("-" * 65)
for name in detectors:
    r = get_row(name)
    print(f"{name:<22} | {float(r['Score']):<10.2f} | {int(r['TP']):<6} | {int(r['FP']):<6} | {int(r['FN']):<6} | {int(r['TN']):<7}")
