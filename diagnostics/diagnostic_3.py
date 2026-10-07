import pandas as pd
import json
import os
from datetime import datetime

def _parse_ts(ts):
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")

def run():
    nab_root = r"c:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab"
    labels_file = os.path.join(nab_root, "labels", "combined_windows.json")
    
    with open(labels_file, "r", encoding="utf-8") as f:
        raw_windows = json.load(f)
        
    ventanas = {}
    for k, v in raw_windows.items():
        ventanas[k.replace("\\", "/")] = [(_parse_ts(w[0]), _parse_ts(w[1])) for w in v]
        
    categories = sorted(list(set([k.split('/')[0] for k in ventanas.keys()])))
    
    total_pts = 0
    total_anomalies = 0
    cat_pts = {c: 0 for c in categories}
    cat_anomalies = {c: 0 for c in categories}
    
    thresholds = [0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]
    
    results = {
        'seasonal': {'pts_ge': {t: 0 for t in thresholds}, 'anom_ge': {t: 0 for t in thresholds}},
        'random': {'pts_ge': {t: 0 for t in thresholds}, 'anom_ge': {t: 0 for t in thresholds}}
    }
    
    cat_zscore_099 = {c: {'pts_ge': 0, 'anom_ge': 0} for c in categories}
    
    zscore_unique_ge_099 = set()
    exact_1_count = 0
    alignment_checks = []
    
    for rel_path, windows in ventanas.items():
        category = rel_path.split('/')[0]
        filename = rel_path.split('/')[1]
        
        z_path = os.path.join(nab_root, "results", "sixeyes_seasonal", category, f"sixeyes_seasonal_{filename}")
        r_path = os.path.join(nab_root, "results", "sixeyes_random", category, f"sixeyes_random_{filename}")
        
        if not os.path.exists(z_path) or not os.path.exists(r_path):
            continue
            
        df_z = pd.read_csv(z_path)
        df_r = pd.read_csv(r_path)
        
        is_anom_list = []
        for ts_str in df_z['timestamp']:
            ts = _parse_ts(ts_str)
            anom = False
            for ws, we in windows:
                if ws <= ts <= we:
                    anom = True
                    break
            is_anom_list.append(anom)
            
        file_pts = len(df_z)
        anom_cnt = sum(is_anom_list)
        
        total_pts += file_pts
        cat_pts[category] += file_pts
        total_anomalies += anom_cnt
        cat_anomalies[category] += anom_cnt
        
        for i, row in df_z.iterrows():
            z_s = row['anomaly_score']
            r_s = df_r.iloc[i]['anomaly_score']
            anom = is_anom_list[i]
            
            if z_s == 1.0:
                exact_1_count += 1
                
            for t in thresholds:
                if z_s >= t:
                    results['seasonal']['pts_ge'][t] += 1
                    if anom:
                        results['seasonal']['anom_ge'][t] += 1
                if r_s >= t:
                    results['random']['pts_ge'][t] += 1
                    if anom:
                        results['random']['anom_ge'][t] += 1
                        
            if z_s >= 0.99:
                zscore_unique_ge_099.add(z_s)
                cat_zscore_099[category]['pts_ge'] += 1
                if anom:
                    cat_zscore_099[category]['anom_ge'] += 1
                    
        if len(alignment_checks) < 3 and len(windows) > 0:
            ws, we = windows[0]
            max_in = -1.0
            max_out = -1.0
            for i, row in df_z.iterrows():
                ts = _parse_ts(row['timestamp'])
                s = row['anomaly_score']
                if ws <= ts <= we:
                    max_in = max(max_in, s)
                else:
                    max_out = max(max_out, s)
            alignment_checks.append({
                'file': rel_path,
                'window': f"{ws} - {we}",
                'max_in': max_in,
                'max_out': max_out
            })

    print("1. Tasa base de puntos en ventanas de anomalía:")
    print(f"Total: {total_anomalies} / {total_pts} ({(total_anomalies/total_pts*100) if total_pts else 0:.2f}%)")
    for c in categories:
        if cat_pts[c] > 0:
            print(f" - {c}: {cat_anomalies[c]} / {cat_pts[c]} ({cat_anomalies[c]/cat_pts[c]*100:.2f}%)")
        
    print("\n2. Puntos >= T y precisión (Puntos en ventana / Puntos >= T):")
    for t in thresholds:
        print(f" Umbral {t}:")
        z_pts = results['seasonal']['pts_ge'][t]
        z_anom = results['seasonal']['anom_ge'][t]
        z_prec = (z_anom/z_pts*100) if z_pts else 0
        r_pts = results['random']['pts_ge'][t]
        r_anom = results['random']['anom_ge'][t]
        r_prec = (r_anom/r_pts*100) if r_pts else 0
        print(f"   - seasonal : {z_pts:5d} pts, {z_anom:5d} en ventana ({z_prec:5.2f}%)")
        print(f"   - random    : {r_pts:5d} pts, {r_anom:5d} en ventana ({r_prec:5.2f}%)")
        
    print(f"\n3. Puntos con score == 1.0 exacto en seasonal: {exact_1_count}")

        
    print("\n3. Desglose por categoría para seasonal en T=0.99:")
    for c in categories:
        pts = cat_zscore_099[c]['pts_ge']
        anoms = cat_zscore_099[c]['anom_ge']
        prec = (anoms/pts*100) if pts else 0
        print(f" - {c}: {pts} pts >= 0.99, {anoms} en ventana ({prec:.2f}%)")
        
    print(f"\n4. Valores de score distintos con score >= 0.99 en zscore_v2: {len(zscore_unique_ge_099)}")
    
    print("\n5. Verificación de alineación (3 ventanas de ejemplo):")
    for idx, check in enumerate(alignment_checks):
        print(f" Ejemplo {idx+1}: {check['file']}")
        print(f"   Ventana: {check['window']}")
        print(f"   Score máx DENTRO: {check['max_in']:.6f}")
        print(f"   Score máx FUERA : {check['max_out']:.6f}")

if __name__ == '__main__':
    run()
