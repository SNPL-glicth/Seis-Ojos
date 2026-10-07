import os
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAB_ROOT = os.path.join(PROJECT_ROOT, "benchmarks", "nab")
RESULTS_DIR = os.path.join(NAB_ROOT, "results")
DATA_DIR = os.path.join(NAB_ROOT, "data")
LABELS_FILE = os.path.join(NAB_ROOT, "labels", "combined_windows.json")

# 1. zscore_v3 y seasonal: las 6 ventanas con T=0.9999
def load_windows():
    with open(LABELS_FILE, "r") as f:
        return json.load(f)

windows_dict = load_windows()

def get_covered_windows(detector, threshold):
    covered = []
    det_dir = os.path.join(RESULTS_DIR, detector)
    if not os.path.exists(det_dir):
        return []
    
    for category in os.listdir(det_dir):
        cat_dir = os.path.join(det_dir, category)
        if not os.path.isdir(cat_dir): continue
        for file in os.listdir(cat_dir):
            if not file.endswith(".csv"): continue
            file_path = os.path.join(cat_dir, file)
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            orig_path = f"{category}/{file.replace(detector + '_', '')}"
            if orig_path not in windows_dict:
                continue
            
            wins = windows_dict[orig_path]
            for i, w in enumerate(wins):
                w_start = pd.to_datetime(w[0])
                w_end = pd.to_datetime(w[1])
                in_w = df[(df["timestamp"] >= w_start) & (df["timestamp"] <= w_end)]
                if (in_w["anomaly_score"] >= threshold).any():
                    covered.append(f"{orig_path} (W{i+1})")
    return covered

cov_zscore = get_covered_windows("sixeyes_zscore_v3", 0.9999)
cov_seasonal = get_covered_windows("sixeyes_seasonal", 0.9999)
print("1. Ventanas cubiertas con T=0.9999")
print(f"zscore_v3 ({len(cov_zscore)}):", sorted(cov_zscore))
print(f"seasonal ({len(cov_seasonal)}):", sorted(cov_seasonal))
print("¿Son las mismas?", sorted(cov_zscore) == sorted(cov_seasonal))
print()

# 2. seasonal: archivos >= 0.5, >= 0.99, >= 0.9999 y top 10 >= 0.99998
seasonal_files_05 = 0
seasonal_files_099 = 0
seasonal_files_09999 = 0
top_files = []

det_dir = os.path.join(RESULTS_DIR, "sixeyes_seasonal")
total_files = 0
if os.path.exists(det_dir):
    for category in os.listdir(det_dir):
        cat_dir = os.path.join(det_dir, category)
        if not os.path.isdir(cat_dir): continue
        for file in os.listdir(cat_dir):
            if not file.endswith(".csv"): continue
            total_files += 1
            file_path = os.path.join(cat_dir, file)
            df = pd.read_csv(file_path)
            orig_path = f"{category}/{file.replace('sixeyes_seasonal_', '')}"
            
            has_05 = (df["anomaly_score"] >= 0.5).any()
            has_099 = (df["anomaly_score"] >= 0.99).any()
            has_09999 = (df["anomaly_score"] >= 0.9999).any()
            pts_extreme = (df["anomaly_score"] >= 0.99998).sum()
            
            if has_05: seasonal_files_05 += 1
            if has_099: seasonal_files_099 += 1
            if has_09999: seasonal_files_09999 += 1
            top_files.append((orig_path, pts_extreme))

top_files.sort(key=lambda x: x[1], reverse=True)
print("2. Archivos en seasonal:")
print(f">= 0.5: {seasonal_files_05}/{total_files}")
print(f">= 0.99: {seasonal_files_099}/{total_files}")
print(f">= 0.9999: {seasonal_files_09999}/{total_files}")
print("Top 10 archivos con puntos >= 0.99998:")
for f, count in top_files[:10]:
    if count > 0:
        print(f"  {f}: {count} puntos")
print()

# 3. For random, zscore_v3, seasonal and each profile:
# get threshold from config/thresholds.json, windows covered, FPs, norm score.
print("3. Métricas en el umbral optimizado:")
with open(os.path.join(NAB_ROOT, "config", "thresholds.json")) as f:
    thresh_data = json.load(f)
with open(os.path.join(RESULTS_DIR, "final_results.json")) as f:
    norm_data = json.load(f)

detectors = ["sixeyes_random", "sixeyes_zscore_v3", "sixeyes_seasonal"]
profiles = ["standard", "reward_low_FP_rate", "reward_low_FN_rate"]

for det in detectors:
    print(f"\nDetector: {det}")
    for prof in profiles:
        try:
            t = thresh_data[det][prof]["threshold"]
            n_score = norm_data.get(det, {}).get(prof, 0)
            if det == "sixeyes_random":
                n_score = norm_data["random"].get(prof, 0)
            elif det == "sixeyes_seasonal":
                n_score = norm_data["sixeyes"].get(f"seasonal_{prof}", 0)
            elif det == "sixeyes_zscore_v3":
                n_score = norm_data.get("sixeyes_zscore_v3", {}).get(prof, 0)
        except Exception as e:
            t = 0
            n_score = 0
            
        wins_covered = 0
        fp_points = 0
        det_dir = os.path.join(RESULTS_DIR, det)
        if os.path.exists(det_dir):
            for category in os.listdir(det_dir):
                cat_dir = os.path.join(det_dir, category)
                if not os.path.isdir(cat_dir): continue
                for file in os.listdir(cat_dir):
                    if not file.endswith(".csv"): continue
                    file_path = os.path.join(cat_dir, file)
                    df = pd.read_csv(file_path, parse_dates=["timestamp"])
                    orig_path = f"{category}/{file.replace(det + '_', '')}"
                    if orig_path not in windows_dict: continue
                    
                    wins = windows_dict[orig_path]
                    
                    is_in_win = pd.Series(False, index=df.index)
                    for w in wins:
                        w_start = pd.to_datetime(w[0])
                        w_end = pd.to_datetime(w[1])
                        win_mask = (df["timestamp"] >= w_start) & (df["timestamp"] <= w_end)
                        is_in_win = is_in_win | win_mask
                        
                        if (df[win_mask]["anomaly_score"] >= t).any():
                            wins_covered += 1
                            
                    fp_points += (df[~is_in_win]["anomaly_score"] >= t).sum()
                    
        print(f"  Perfil {prof}:")
        print(f"    Umbral: {t:.5f}")
        print(f"    Ventanas cubiertas: {wins_covered}/116")
        print(f"    Puntos fuera de ventana (>= T): {fp_points}")
        print(f"    Puntaje Normalizado (NAB): {n_score:.2f}")

print()

# 5. Test anti-fuga real para el seasonal
import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))); from src.detectors.seasonal import DailySeasonalRobustZ
print("5. Test anti-fuga causal")
det_fuga = DailySeasonalRobustZ()
scores_normal = []
import datetime
ts = datetime.datetime(2026, 10, 7, 0, 0, 0)
for i in range(100):
    scores_normal.append(det_fuga.update(10.0, ts))
    ts += datetime.timedelta(minutes=5)

det_alt = DailySeasonalRobustZ()
scores_alt = []
ts = datetime.datetime(2026, 10, 7, 0, 0, 0)
for i in range(100):
    val = 10.0 if i < 50 else 999.9  # Modify from 50 onwards
    scores_alt.append(det_alt.update(val, ts))
    ts += datetime.timedelta(minutes=5)

diff = False
for i in range(50):
    if scores_normal[i] != scores_alt[i]:
         diff = True
print(f"¿Los scores hasta k=50 son idénticos a pesar de modificar el futuro?: {not diff}")

