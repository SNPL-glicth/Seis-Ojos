"""Evaluación comparativa controlada sobre TSB-AD Tuning-43.

Compara punto a punto bajo el protocolo oficial de TSB-AD:
  1. Baseline: ESN + PercentileCalibrator + StreamingPeakDecay
  2. Fase III: ContinuousPlasticityController + PercentileCalibrator + StreamingPeakDecay
  3. Fase V:   RegimeAwarePlasticityController (Huber + Memoria K=5) + PercentileCalibrator + StreamingPeakDecay

Métricas evaluadas: VUS-PR, AUC-PR, AUC-ROC, tiempo de cómputo por serie.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay, DEFAULT_K_WINDOW, DEFAULT_TAU
from sixeyes.experimental.continuous_plasticity_controller import ContinuousPlasticityController
from sixeyes.experimental.regime_aware_controller import RegimeAwarePlasticityController
from sixeyes.evaluation.tsb_runner import run_single_series


def resolve_paths() -> tuple[Path, Path]:
    candidates = [
        ROOT_DIR / "benchmarks" / "tsb_ad" / "TSB-AD",
        ROOT_DIR.parent / "benchmarks" / "tsb_ad" / "TSB-AD",
    ]
    for base in candidates:
        lf = base / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
        dd = base / "Datasets" / "TSB-AD-U"
        if lf.is_file() and dd.is_dir():
            return lf, dd
    raise FileNotFoundError("No se encontró el dataset TSB-AD Tuning.")


def get_factories(k_window: int = DEFAULT_K_WINDOW, tau: float = DEFAULT_TAU,
                  cal_window: int = 16000, seed: int = 42) -> Dict[str, Callable[[], Any]]:
    return {
        "baseline_esn": lambda: StreamingPeakDecay(
            PercentileCalibrator(EchoStatePredictor(seed=seed), max_history=cal_window),
            k_window=k_window, tau=tau
        ),
        "fase_iii_plasticity": lambda: StreamingPeakDecay(
            PercentileCalibrator(
                ContinuousPlasticityController(enable_controller=True, seed=seed),
                max_history=cal_window
            ),
            k_window=k_window, tau=tau
        ),
        "fase_v_regime_huber": lambda: StreamingPeakDecay(
            PercentileCalibrator(
                RegimeAwarePlasticityController(
                    enable_controller=True,
                    enable_memory=True,
                    scale_estimator_type="huber",
                    quarantine_cap=0.05,
                    quarantine_steps=250,
                    seed=seed,
                ),
                max_history=cal_window
            ),
            k_window=k_window, tau=tau
        ),
    }


def main():
    start_global = time.time()
    list_file, data_dir = resolve_paths()
    df_list = pd.read_csv(list_file)
    files_to_run = df_list[~df_list['file_name'].str.contains('_NAB_')]['file_name'].tolist()

    print(f"============================================================")
    print(f"Iniciando evaluación comparativa TSB-AD Tuning ({len(files_to_run)} series)")
    print(f"============================================================")

    factories = get_factories()
    results_by_var: Dict[str, List[Dict[str, Any]]] = {k: [] for k in factories}

    for idx, f in enumerate(files_to_run):
        dataset = f.split('_')[1]
        file_path = data_dir / f
        print(f"\n[{idx+1}/{len(files_to_run)}] Serie: {f} (dataset: {dataset})")

        for var_name, factory in factories.items():
            res = run_single_series(factory, file_path, timeout_s=180.0)
            if res["status"] == "OK":
                m = res["metrics"]
                row = {
                    "serie": f,
                    "dataset": dataset,
                    "variante": var_name,
                    "longitud": res["length"],
                    "VUS-PR": m.get("VUS-PR"),
                    "AUC-PR": m.get("AUC-PR"),
                    "AUC-ROC": m.get("AUC-ROC"),
                    "tiempo_det_s": res["time_detector"],
                    "tiempo_met_s": res["time_metrics"],
                    "status": "OK",
                }
                print(f"   * {var_name:20s}: VUS-PR={m.get('VUS-PR', 0.0):.4f} | AUC-PR={m.get('AUC-PR', 0.0):.4f} | det_time={res['time_detector']:.2f}s")
            else:
                row = {
                    "serie": f,
                    "dataset": dataset,
                    "variante": var_name,
                    "longitud": None,
                    "VUS-PR": None,
                    "AUC-PR": None,
                    "AUC-ROC": None,
                    "tiempo_det_s": None,
                    "tiempo_met_s": None,
                    "status": f"ERROR: {res.get('error')}",
                }
                print(f"   * {var_name:20s}: FALLÓ ({res.get('error')})")
            results_by_var[var_name].append(row)

    # Consolidación
    all_rows = []
    for var_name, rows in results_by_var.items():
        all_rows.extend(rows)

    df_res = pd.DataFrame(all_rows)
    out_dir = ROOT_DIR / "results" / "diagnostics_plasticity"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "tsb_ad_tuning_comparison.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\nResultados detallados guardados en: {csv_out}")

    print("\n" + "=" * 60)
    print("RESUMEN GLOBAL TSB-AD TUNING (Promedio por Serie):")
    print("=" * 60)
    summary_dict = {}
    for var_name in factories:
        sub = df_res[(df_res["variante"] == var_name) & (df_res["status"] == "OK")]
        vus_mean = sub["VUS-PR"].mean()
        auc_pr_mean = sub["AUC-PR"].mean()
        auc_roc_mean = sub["AUC-ROC"].mean()
        time_det_mean = sub["tiempo_det_s"].mean()
        print(f"Variante: {var_name:20s}")
        print(f"  * Series evaluadas : {len(sub)}/{len(files_to_run)}")
        print(f"  * VUS-PR medio     : {vus_mean:.4f}")
        print(f"  * AUC-PR medio     : {auc_pr_mean:.4f}")
        print(f"  * AUC-ROC medio    : {auc_roc_mean:.4f}")
        print(f"  * Tiempo det medio : {time_det_mean:.3f} s/serie")
        summary_dict[var_name] = {
            "n_eval": len(sub),
            "vus_pr_mean": float(vus_mean) if not math.isnan(vus_mean) else None,
            "auc_pr_mean": float(auc_pr_mean) if not math.isnan(auc_pr_mean) else None,
            "auc_roc_mean": float(auc_roc_mean) if not math.isnan(auc_roc_mean) else None,
            "time_det_mean": float(time_det_mean) if not math.isnan(time_det_mean) else None,
        }

    json_out = out_dir / "tsb_ad_tuning_comparison.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4)
    print(f"Resumen JSON guardado en: {json_out}")
    print(f"Tiempo total transcurrido: {time.time() - start_global:.2f} s")


if __name__ == "__main__":
    main()
