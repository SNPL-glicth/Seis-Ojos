import pandas as pd
import glob
import os
import sys

def analyze(name, pattern):
    files = glob.glob(pattern)
    if not files:
        print(f"No se encontraron archivos para {name}")
        return

    total_points = 0
    c_0 = 0
    c_05 = 0
    c_09 = 0
    c_099 = 0
    c_1 = 0
    nan_count = 0
    unique_scores = set()
    max_decimals = 0
    min_decimals = 9999

    for f in files:
        df = pd.read_csv(f)
        scores = df['anomaly_score']
        nan_count += scores.isna().sum()
        scores = scores.dropna()
        
        total_points += len(scores)
        c_0 += (scores == 0.0).sum()
        c_05 += (scores >= 0.5).sum()
        c_09 += (scores >= 0.9).sum()
        c_099 += (scores >= 0.99).sum()
        c_1 += (scores == 1.0).sum()
        
        unique_scores.update(scores.unique())
        
        df_str = pd.read_csv(f, dtype={'anomaly_score': str})
        for s in df_str['anomaly_score'].dropna():
            if '.' in s:
                dec = len(s.split('.')[1])
                if dec > max_decimals: max_decimals = dec
                if dec < min_decimals: min_decimals = dec
            else:
                min_decimals = 0

    print(f"--- REPORTE {name} ---")
    print(f"Archivos: {len(files)}")
    print(f"Puntos totales: {total_points}")
    print(f"NaN / Vacíos: {nan_count}")
    print(f"== 0.0 : {c_0} ({c_0 / total_points * 100:.2f}%)")
    print(f">= 0.5 : {c_05} ({c_05 / total_points * 100:.2f}%)")
    print(f">= 0.9 : {c_09} ({c_09 / total_points * 100:.2f}%)")
    print(f">= 0.99: {c_099} ({c_099 / total_points * 100:.2f}%)")
    print(f"== 1.0 : {c_1} ({c_1 / total_points * 100:.2f}%)")
    print(f"Scores únicos: {len(unique_scores)}")
    if min_decimals == 9999: min_decimals = 0
    print(f"Decimales: {min_decimals} a {max_decimals}\n")

if __name__ == '__main__':
    base_dir = r"c:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab\results"
    analyze("sixeyes_zscore", os.path.join(base_dir, "sixeyes_zscore", "*", "*.csv"))
    analyze("sixeyes_random", os.path.join(base_dir, "sixeyes_random", "*", "*.csv"))
