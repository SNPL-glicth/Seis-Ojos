import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # type: ignore

# Asegurar importación de src
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(root, "src"))

from detectors.calibrador_percentil import PercentileCalibrator
from detectors.zscore_robusto import RollingRobustZ
from detectors.subespacio import SubspaceResidualDetector

def auditar_serie(file_name, file_path):
    df = pd.read_csv(file_path).dropna()
    values = df.iloc[:, 0].values.astype(float)
    labels = df['Label'].astype(int).values
    n = len(values)

    det_z = PercentileCalibrator(RollingRobustZ())
    det_sub = PercentileCalibrator(SubspaceResidualDetector())

    scores_z = np.zeros(n, dtype=float)
    scores_sub = np.zeros(n, dtype=float)

    for i in range(n):
        val = values[i]
        scores_z[i] = det_z.update(val, str(i))
        scores_sub[i] = det_sub.update(val, str(i))

    norm_mask = (labels == 0)
    anom_mask = (labels == 1)

    # 1. Dominancia estocástica
    # Global
    dom_z_global = np.mean(scores_z > scores_sub)
    dom_sub_global = np.mean(scores_sub > scores_z)
    tie_global = np.mean(scores_z == scores_sub)

    # En normales
    dom_z_norm = np.mean(scores_z[norm_mask] > scores_sub[norm_mask]) if norm_mask.any() else np.nan
    dom_sub_norm = np.mean(scores_sub[norm_mask] > scores_z[norm_mask]) if norm_mask.any() else np.nan
    tie_norm = np.mean(scores_z[norm_mask] == scores_sub[norm_mask]) if norm_mask.any() else np.nan

    # En anomalías
    dom_z_anom = np.mean(scores_z[anom_mask] > scores_sub[anom_mask]) if anom_mask.any() else np.nan
    dom_sub_anom = np.mean(scores_sub[anom_mask] > scores_z[anom_mask]) if anom_mask.any() else np.nan
    tie_anom = np.mean(scores_z[anom_mask] == scores_sub[anom_mask]) if anom_mask.any() else np.nan

    # 2. Distribución en normales
    sz_norm = scores_z[norm_mask]
    ssub_norm = scores_sub[norm_mask]

    sz_norm_stats = {
        "mean": np.mean(sz_norm), "med": np.median(sz_norm),
        "p90": np.percentile(sz_norm, 90), "p95": np.percentile(sz_norm, 95), "p99": np.percentile(sz_norm, 99),
        "zero_pct": np.mean(sz_norm == 0.0)
    }
    ssub_norm_stats = {
        "mean": np.mean(ssub_norm), "med": np.median(ssub_norm),
        "p90": np.percentile(ssub_norm, 90), "p95": np.percentile(ssub_norm, 95), "p99": np.percentile(ssub_norm, 99),
        "zero_pct": np.mean(ssub_norm == 0.0)
    }

    # 3. Distribución en anomalías
    if anom_mask.any():
        sz_anom = scores_z[anom_mask]
        ssub_anom = scores_sub[anom_mask]
        sz_anom_stats = {
            "mean": np.mean(sz_anom), "med": np.median(sz_anom),
            "p90": np.percentile(sz_anom, 90), "p95": np.percentile(sz_anom, 95)
        }
        ssub_anom_stats = {
            "mean": np.mean(ssub_anom), "med": np.median(ssub_anom),
            "p90": np.percentile(ssub_anom, 90), "p95": np.percentile(ssub_anom, 95)
        }
    else:
        sz_anom_stats = ssub_anom_stats = {}

    # 4. Correlación de Spearman en espacio normal
    if norm_mask.sum() > 2:
        rho_norm, _ = spearmanr(sz_norm, ssub_norm)
    else:
        rho_norm = np.nan

    # 5. Deriva temporal de falsos positivos en normales (por cuartiles)
    # Tasa de scores > 0.90 en puntos normales a lo largo del tiempo
    n_q = n // 4
    drift_z = []
    drift_sub = []
    for q in range(4):
        idx_start = q * n_q
        idx_end = (q + 1) * n_q if q < 3 else n
        q_norm = norm_mask[idx_start:idx_end]
        if q_norm.any():
            rate_z = np.mean(scores_z[idx_start:idx_end][q_norm] > 0.90)
            rate_sub = np.mean(scores_sub[idx_start:idx_end][q_norm] > 0.90)
        else:
            rate_z, rate_sub = 0.0, 0.0
        drift_z.append(rate_z)
        drift_sub.append(rate_sub)

    return {
        "file": file_name, "n": n, "anom_count": int(anom_mask.sum()),
        "dom_z_global": dom_z_global, "dom_sub_global": dom_sub_global, "tie_global": tie_global,
        "dom_z_norm": dom_z_norm, "dom_sub_norm": dom_sub_norm, "tie_norm": tie_norm,
        "dom_z_anom": dom_z_anom, "dom_sub_anom": dom_sub_anom, "tie_anom": tie_anom,
        "sz_norm_stats": sz_norm_stats, "ssub_norm_stats": ssub_norm_stats,
        "sz_anom_stats": sz_anom_stats, "ssub_anom_stats": ssub_anom_stats,
        "rho_norm": rho_norm,
        "drift_z": drift_z, "drift_sub": drift_sub
    }

def main():
    list_file = r"..\benchmarks\tsb_ad\TSB-AD\Datasets\File_List\TSB-AD-U-Tuning.csv"
    data_dir = r"..\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U"

    df_list = pd.read_csv(list_file)
    files = df_list[~df_list['file_name'].str.contains('_NAB_')]['file_name'].tolist()

    smd_files = [f for f in files if "_SMD_" in f]
    yahoo_files = [f for f in files if "_YAHOO_" in f]

    targets = [("SMD", smd_files), ("YAHOO", yahoo_files)]

    for ds_name, file_list in targets:
        print(f"\n=======================================================")
        print(f" AUDITORÍA DE CALIBRACIÓN: {ds_name} (n={len(file_list)} series)")
        print(f"=======================================================")

        results = []
        for f in file_list:
            fp = os.path.join(data_dir, f)
            res = auditar_serie(f, fp)
            results.append(res)
            print(f"Procesada: {f}")

        # Resumen de dominancia
        print(f"\n--- 1. TASA DE DOMINANCIA / SELECCIÓN ({ds_name}) ---")
        rows_dom = []
        for r in results:
            rows_dom.append({
                "archivo": r["file"][:30],
                "N": r["n"],
                "Dom Z (Global)": f"{r['dom_z_global']*100:.1f}%",
                "Dom Sub (Global)": f"{r['dom_sub_global']*100:.1f}%",
                "Empate": f"{r['tie_global']*100:.1f}%",
                "Dom Z (Normal)": f"{r['dom_z_norm']*100:.1f}%",
                "Dom Sub (Normal)": f"{r['dom_sub_norm']*100:.1f}%",
                "Dom Z (Anomalía)": f"{r['dom_z_anom']*100:.1f}%" if not np.isnan(r['dom_z_anom']) else "N/A",
                "Dom Sub (Anomalía)": f"{r['dom_sub_anom']*100:.1f}%" if not np.isnan(r['dom_sub_anom']) else "N/A",
            })
        df_dom = pd.DataFrame(rows_dom)
        print(df_dom.to_string(index=False))

        # Medias por dataset
        mean_dom_z_glob = np.mean([r['dom_z_global'] for r in results])
        mean_dom_sub_glob = np.mean([r['dom_sub_global'] for r in results])
        mean_dom_z_norm = np.mean([r['dom_z_norm'] for r in results])
        mean_dom_sub_norm = np.mean([r['dom_sub_norm'] for r in results])
        print(f"\nPROMEDIOS {ds_name}:")
        print(f"  Dominancia Global:      ZScore = {mean_dom_z_glob*100:.2f}% | Subespacio = {mean_dom_sub_glob*100:.2f}%")
        print(f"  Dominancia en Normales: ZScore = {mean_dom_z_norm*100:.2f}% | Subespacio = {mean_dom_sub_norm*100:.2f}%")

        # Resumen de distribución en normales
        print(f"\n--- 2. DISTRIBUCIÓN DE SCORES EN TIMESTEPS NORMALES ({ds_name}) ---")
        rows_dist = []
        for r in results:
            sz = r["sz_norm_stats"]
            ssub = r["ssub_norm_stats"]
            rows_dist.append({
                "archivo": r["file"][:25],
                "Z_med": f"{sz['med']:.3f}", "Sub_med": f"{ssub['med']:.3f}",
                "Z_p90": f"{sz['p90']:.3f}", "Sub_p90": f"{ssub['p90']:.3f}",
                "Z_p95": f"{sz['p95']:.3f}", "Sub_p95": f"{ssub['p95']:.3f}",
                "Z_zeros": f"{sz['zero_pct']*100:.1f}%", "Sub_zeros": f"{ssub['zero_pct']*100:.1f}%"
            })
        print(pd.DataFrame(rows_dist).to_string(index=False))

        # Resumen en anomalías
        print(f"\n--- 3. DISTRIBUCIÓN DE SCORES EN TIMESTEPS ANÓMALOS ({ds_name}) ---")
        rows_anom = []
        for r in results:
            if r["sz_anom_stats"]:
                sz = r["sz_anom_stats"]
                ssub = r["ssub_anom_stats"]
                rows_anom.append({
                    "archivo": r["file"][:25],
                    "Anom_pts": r["anom_count"],
                    "Z_med": f"{sz['med']:.3f}", "Sub_med": f"{ssub['med']:.3f}",
                    "Z_p90": f"{sz['p90']:.3f}", "Sub_p90": f"{ssub['p90']:.3f}"
                })
        print(pd.DataFrame(rows_anom).to_string(index=False))

        # Correlación de Spearman en normales
        print(f"\n--- 4. CORRELACIÓN DE SPEARMAN EN ESPACIO NORMAL ({ds_name}) ---")
        for r in results:
            print(f"  {r['file'][:35]}: Spearman rho = {r['rho_norm']:.4f}")
        print(f"  Promedio Spearman rho: {np.nanmean([r['rho_norm'] for r in results]):.4f}")

        # Deriva temporal (falsos positivos > 0.90 por cuartil)
        print(f"\n--- 5. DERIVA TEMPORAL: Tasa de FP (> 0.90 en normales) por Cuartil ({ds_name}) ---")
        for r in results:
            q_z_str = " -> ".join([f"{v*100:.2f}%" for v in r["drift_z"]])
            q_sub_str = " -> ".join([f"{v*100:.2f}%" for v in r["drift_sub"]])
            print(f"  {r['file'][:30]}:")
            print(f"    Z-Score FP rate (Q1..Q4):    {q_z_str}")
            print(f"    Subespacio FP rate (Q1..Q4): {q_sub_str}")

if __name__ == "__main__":
    main()
