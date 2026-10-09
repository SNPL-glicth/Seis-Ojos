import pandas as pd
from pathlib import Path

csv_path = Path("../benchmarks/nab/results/sixeyes_peak_decay/sixeyes_peak_decay_standard_scores.csv")
df = pd.read_csv(csv_path)

# Filter out Totals
files_df = df[df["Detector"] != "Totals"].copy()

print(f"Total datasets evaluados: {len(files_df)}")

# Distribution of scores
positive = files_df[files_df["Score"] > 0]
zero = files_df[files_df["Score"] == 0]
negative = files_df[files_df["Score"] < 0]

print(f"Datasets con score > 0 (Detección positiva limpia): {len(positive)} ({100*len(positive)/len(files_df):.1f}%)")
print(f"Datasets con score == 0 (Sin alarmas espurias / neutros): {len(zero)} ({100*len(zero)/len(files_df):.1f}%)")
print(f"Datasets con score < 0 (Penalizaciones netas): {len(negative)} ({100*len(negative)/len(files_df):.1f}%)")

# Extract category
files_df["Category"] = files_df["File"].apply(lambda x: x.split("/")[0] if "/" in x else "root")

print("\n--- Desglose por Categoría en NAB ---")
cat_summary = files_df.groupby("Category").agg(
    Files=("File", "count"),
    Mean_Score=("Score", "mean"),
    Total_Score=("Score", "sum"),
    Total_TP=("TP", "sum"),
    Total_FP=("FP", "sum"),
    Total_FN=("FN", "sum"),
)
print(cat_summary.to_string())

totals_row = df[df["Detector"] == "Totals"].iloc[0]
print("\n--- Totales Globales del Corpus ---")
print(f"Score Bruto Total: {totals_row['Score']:.4f}")
print(f"TP Brutos: {totals_row['TP']}")
print(f"FP Brutos: {totals_row['FP']}")
print(f"FN Brutos: {totals_row['FN']}")
print(f"TN Brutos: {totals_row['TN']}")
print(f"Total Registros Evaluados: {totals_row['Total_Count']}")
