"""Evaluación Comparativa Inmutable: P05 Campeón vs P05 con Fase V (Tuning-43).

Compara directamente bajo el arnés oficial de corral_t43_predictive:
  - Brazo P05 (Campeón Histórico): SpatialSpikeBypass(PercentileCalibrator(PredictiveReadoutESN(alpha=0.5)))
  - Brazo P05_FaseV:               SpatialSpikeBypass(PercentileCalibrator(RegimeAwarePlasticityController(alpha=0.5, Huber, Memoria K=5)))
  - Brazo P05_FaseIII:             SpatialSpikeBypass(PercentileCalibrator(ContinuousPlasticityController(alpha=0.5)))

El criterio de éxito no busca un número cosmético:
Calcula la diferencia pareada exacta por serie (Delta = FaseV - P05)
con Bootstrap al 95% de 4.000 iteraciones pareadas.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.experimental.continuous_plasticity_controller import ContinuousPlasticityController
from sixeyes.experimental.regime_aware_controller import RegimeAwarePlasticityController

TSB_DIR = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets")
OUT_CSV = ROOT_DIR / "results" / "diagnostics_plasticity" / "p05_vs_fase_v_t43.csv"
OUT_JSON = ROOT_DIR / "results" / "diagnostics_plasticity" / "p05_vs_fase_v_t43.json"


def factory_arm(arm: str):
    if arm == "P05_FaseV":
        rac = RegimeAwarePlasticityController(
            seed=42,
            alpha_recon=0.5,
            enable_controller=True,
            enable_memory=True,
            scale_estimator_type="huber",
            quarantine_cap=0.05,
            quarantine_steps=250,
        )
        cal = PercentileCalibrator(rac, max_history=16000)
        return SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)

    if arm == "P05_FaseIII":
        cpc = ContinuousPlasticityController(
            seed=42,
            alpha_recon=0.5,
            enable_controller=True,
        )
        cal = PercentileCalibrator(cpc, max_history=16000)
        return SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)

    raise ValueError(f"Brazo desconocido: {arm}")


def main():
    start_time = time.time()
    # 1. Cargar datos existentes de P05 y P10 desde corral_t43_predictive.csv
    corral_csv = ROOT_DIR / "results" / "corral_t43_predictive.csv"
    if not corral_csv.is_file():
        raise FileNotFoundError(f"No se encontró {corral_csv}")

    df_corral = pd.read_csv(corral_csv)
    df_p05 = df_corral[df_corral["arm"] == "P05"].copy()
    assert len(df_p05) == 43, f"Se esperaban 43 series de P05, hay {len(df_p05)}"

    list_file = TSB_DIR / "File_List" / "TSB-AD-U-Tuning.csv"
    lst = pd.read_csv(list_file)
    files = [f for f in lst["file_name"].tolist() if "_NAB_" not in f]
    assert len(files) == 43

    new_arms = ["P05_FaseV", "P05_FaseIII"]
    print("=" * 60, flush=True)
    print(f"Iniciando evaluación controlada de P05 vs Fase V en Tuning-43", flush=True)
    print(f"Brazos a evaluar: {new_arms}", flush=True)
    print("=" * 60, flush=True)

    rows = []
    # Cargar filas previas de P05 y P10 para mantener paridad
    for _, r in df_corral[df_corral["arm"].isin(["P05", "P10"])].iterrows():
        rows.append({
            "arm": r["arm"],
            "file": r["file"],
            "status": r["status"],
            "AUC-ROC": float(r["AUC-ROC"]),
            "AUC-PR": float(r["AUC-PR"]),
            "VUS-PR": float(r["VUS-PR"]),
            "VUS-ROC": float(r["VUS-ROC"]),
        })

    for i, f in enumerate(files):
        data_file = TSB_DIR / "TSB-AD-U" / f
        for arm in new_arms:
            t0 = time.time()
            try:
                res = run_single_series(lambda a=arm: factory_arm(a), data_file, timeout_s=180.0)
                m = res.get("metrics", {})
                st = res.get("status", "ERROR")
                row = {
                    "arm": arm,
                    "file": f,
                    "status": st,
                    "AUC-ROC": float(m.get("AUC-ROC", np.nan)) if st == "OK" else np.nan,
                    "AUC-PR": float(m.get("AUC-PR", np.nan)) if st == "OK" else np.nan,
                    "VUS-PR": float(m.get("VUS-PR", np.nan)) if st == "OK" else np.nan,
                    "VUS-ROC": float(m.get("VUS-ROC", np.nan)) if st == "OK" else np.nan,
                }
                print(f"[{i+1}/43] {arm} en {f[:30]}...: status={st} | AUC-ROC={row['AUC-ROC']:.4f} | VUS-PR={row['VUS-PR']:.4f} ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                row = {
                    "arm": arm,
                    "file": f,
                    "status": f"ERROR:{type(e).__name__}",
                    "AUC-ROC": np.nan,
                    "AUC-PR": np.nan,
                    "VUS-PR": np.nan,
                    "VUS-ROC": np.nan,
                }
                print(f"[{i+1}/43] {arm} en {f[:30]}...: FALLÓ ({e})", flush=True)
            rows.append(row)

    df_out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nResultados exportados a: {OUT_CSV}", flush=True)

    # Análisis pareado riguroso con Bootstrap (4.000 iteraciones)
    ok = df_out[df_out["status"] == "OK"]
    piv_roc = ok.pivot(index="file", columns="arm", values="AUC-ROC")
    piv_vus = ok.pivot(index="file", columns="arm", values="VUS-PR")
    piv_pr = ok.pivot(index="file", columns="arm", values="AUC-PR")

    res_summary = {
        "n_series": len(files),
        "means_auc_roc": {a: round(float(piv_roc[a].mean()), 4) for a in piv_roc.columns if a in piv_roc},
        "means_vus_pr": {a: round(float(piv_vus[a].mean()), 4) for a in piv_vus.columns if a in piv_vus},
        "means_auc_pr": {a: round(float(piv_pr[a].mean()), 4) for a in piv_pr.columns if a in piv_pr},
    }

    rng = np.random.default_rng(42)
    # Comparar P05_FaseV vs P05
    for arm_comp in ["P05_FaseV", "P05_FaseIII"]:
        if arm_comp in piv_roc.columns:
            d_roc = (piv_roc[arm_comp] - piv_roc["P05"]).dropna().to_numpy()
            boot_roc = [rng.choice(d_roc, len(d_roc), replace=True).mean() for _ in range(4000)]
            ic_roc = np.quantile(boot_roc, [0.025, 0.975])

            d_vus = (piv_vus[arm_comp] - piv_vus["P05"]).dropna().to_numpy()
            boot_vus = [rng.choice(d_vus, len(d_vus), replace=True).mean() for _ in range(4000)]
            ic_vus = np.quantile(boot_vus, [0.025, 0.975])

            d_pr = (piv_pr[arm_comp] - piv_pr["P05"]).dropna().to_numpy()
            boot_pr = [rng.choice(d_pr, len(d_pr), replace=True).mean() for _ in range(4000)]
            ic_pr = np.quantile(boot_pr, [0.025, 0.975])

            res_summary[f"{arm_comp}_vs_P05"] = {
                "n_pareadas": int(len(d_roc)),
                "delta_auc_roc": round(float(d_roc.mean()), 4),
                "ic95_auc_roc": [round(float(ic_roc[0]), 4), round(float(ic_roc[1]), 4)],
                "win_rate_auc_roc": round(float((d_roc > 0).mean()), 3),
                "delta_vus_pr": round(float(d_vus.mean()), 4),
                "ic95_vus_pr": [round(float(ic_vus[0]), 4), round(float(ic_vus[1]), 4)],
                "delta_auc_pr": round(float(d_pr.mean()), 4),
                "ic95_auc_pr": [round(float(ic_pr[0]), 4), round(float(ic_pr[1]), 4)],
                "supera_p05": bool(d_roc.mean() > 0 and ic_roc[0] > 0),
            }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res_summary, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("RESUMEN PRE-REGISTRADO P05 vs FASE V:", flush=True)
    print("=" * 60, flush=True)
    print(json.dumps(res_summary, indent=2), flush=True)
    print(f"\nTiempo total: {time.time() - start_time:.1f} s", flush=True)


if __name__ == "__main__":
    main()
