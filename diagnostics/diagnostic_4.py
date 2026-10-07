import json
import os
import pandas as pd
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from detectors.seasonal import DailySeasonalRobustZ
from detectors.zscore_robusto import NORMAL_SCALE_MAD, SCORE_SIN_EVIDENCIA

def _parse_ts(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

def run():
    nab_root = os.path.join("..", "..", "benchmarks", "nab")
    with open(os.path.join(nab_root, "labels", "combined_windows.json"), "r") as f:
        ventanas_raw = json.load(f)
        
    ventanas = {}
    for rel_path, w_list in ventanas_raw.items():
        parsed = []
        for w in w_list:
            parsed.append((_parse_ts(w[0][:19]), _parse_ts(w[1][:19])))
        ventanas[rel_path] = parsed

    thresholds = [0.5, 0.9, 0.99, 0.9999]
    detectors = ['sixeyes_zscore_v3', 'sixeyes_seasonal', 'sixeyes_random']
    
    # metrics: { det: { t: { 'windows_hit': 0, 'latencies': [] } } }
    metrics = {d: {t: {'windows_hit': 0, 'latencies': []} for t in thresholds} for d in detectors}
    total_windows = 0
    
    # Store data for step 5
    high_score_events = []
    
    for rel_path, windows in ventanas.items():
        total_windows += len(windows)
        category = rel_path.split('/')[0]
        filename = rel_path.split('/')[1]
        
        dfs = {}
        for d in detectors:
            path = os.path.join(nab_root, "results", d, category, f"{d}_{filename}")
            if os.path.exists(path):
                dfs[d] = pd.read_csv(path)
            
        if not dfs:
            continue
            
        # Recreate the seasonal detector exactly as in run_nab.py to inspect internal state
        det_seasonal = DailySeasonalRobustZ()

        for w_idx, (ws, we) in enumerate(windows):
            window_size_td = (we - ws).total_seconds()
            if window_size_td <= 0:
                continue
                
            for d in detectors:
                if d not in dfs: continue
                df = dfs[d]
                
                # Filter points in this window
                in_window = []
                for i, row in df.iterrows():
                    ts = _parse_ts(row['timestamp'])
                    if ws <= ts <= we:
                        in_window.append((ts, row['anomaly_score']))
                        
                for t in thresholds:
                    first_hit = next((pts for pts in in_window if pts[1] >= t), None)
                    if first_hit is not None:
                        metrics[d][t]['windows_hit'] += 1
                        latency_sec = (first_hit[0] - ws).total_seconds()
                        latency_frac = latency_sec / window_size_td
                        metrics[d][t]['latencies'].append(latency_frac)
                        
        # Step 5 check
        if 'sixeyes_seasonal' in dfs:
            df = dfs['sixeyes_seasonal']
            for i, row in df.iterrows():
                val = row['value']
                ts = row['timestamp']
                # Predict what the detector will do internally
                
                b_idx = det_seasonal._parse_and_bucket(ts)
                
                # Check internal state of the bucket BEFORE update
                if b_idx in det_seasonal._buckets:
                    robz = det_seasonal._buckets[b_idx]
                    
                    if len(robz._window) >= robz._min_observations:
                        import statistics
                        mediana = statistics.median(robz._window)
                        desviaciones = [abs(x - mediana) for x in robz._window]
                        mad = statistics.median(desviaciones)
                        
                        if mad == 0.0:
                            escala = robz._calcular_escala_alternativa(desviaciones)
                            rama = "MAD=0 Fallback"
                        else:
                            escala = NORMAL_SCALE_MAD * mad
                            rama = "MAD>0 Normal"
                            
                        if escala > 0.0:
                            z = abs(val - mediana) / escala
                        else:
                            z = 0.0
                            rama = "Sin evidencia"
                            
                        n_obs = len(robz._window)
                    else:
                        z = 0.0
                        escala = 0.0
                        rama = "Warmup"
                        n_obs = len(robz._window)
                else:
                    z = 0.0
                    escala = 0.0
                    rama = "Warmup"
                    n_obs = 0
                    
                score = det_seasonal.update(val, ts)
                if score >= 0.99998:
                    high_score_events.append({
                        'z': z,
                        'escala': escala,
                        'rama': rama,
                        'n_obs': n_obs,
                        'score': score
                    })

    print("4. Porcentaje de VENTANAS con al menos un punto con score >= T, y Latencia Media:")
    print(f"Total ventanas: {total_windows}")
    for d in detectors:
        print(f"\n Detector: {d}")
        for t in thresholds:
            hits = metrics[d][t]['windows_hit']
            pct = (hits / total_windows * 100) if total_windows else 0
            lats = metrics[d][t]['latencies']
            mean_lat = (sum(lats) / len(lats)) if lats else 0
            print(f"   Umbral {t:<6}: {pct:5.1f}% ventanas ({hits}/{total_windows}), Latencia media: {mean_lat:.3f}")
            
    print("\n5. Distribución para puntos de seasonal con score >= 0.99998:")
    print(f"Total de puntos analizados: {len(high_score_events)}")
    ramas_count = {}
    for ev in high_score_events:
        ramas_count[ev['rama']] = ramas_count.get(ev['rama'], 0) + 1
        
    print(f"Ramas usadas: {ramas_count}")
    
    # Show stats of z, escala, n_obs
    if high_score_events:
        z_vals = [e['z'] for e in high_score_events]
        s_vals = [e['escala'] for e in high_score_events]
        n_vals = [e['n_obs'] for e in high_score_events]
        import statistics
        print(f"Z      : min={min(z_vals):.1f}, max={max(z_vals):.1f}, median={statistics.median(z_vals):.1f}")
        print(f"Escala : min={min(s_vals):.6f}, max={max(s_vals):.6f}, median={statistics.median(s_vals):.6f}")
        print(f"N_Obs  : min={min(n_vals)}, max={max(n_vals)}, median={statistics.median(n_vals):.1f}")
        
if __name__ == "__main__":
    run()
