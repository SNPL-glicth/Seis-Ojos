"""Runner canónico y paralelo para la corrida congelada de las 472 frescas.

Ejecuta los cuatro brazos pre-registrados:
- P10: Base oficial (alpha_recon=1.0)
- P05: Candidato congelado (alpha_recon=0.5)
- random: Detector uniforme aleatorio (seed=42)
- zscore: RollingRobustZ(window_size=50)

Escribe el CSV de salida usando csv.DictWriter con cabecera exacta:
    series,family,n_points,arm,auc_roc,auc_pr,vus_pr
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from sixeyes.evaluation.tsb_runner import _resolve_tsb_path, _get_tsb_metrics
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN

TSB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets")
TSBU = TSB / "TSB-AD-U"
FL = TSB / "File_List"
ARMS = ("P10", "P05", "random", "zscore")
FIELDNAMES = ["series", "family", "n_points", "arm", "auc_roc", "auc_pr", "vus_pr"]


def factory(arm: str):
    if arm == "P10":
        esn = PredictiveReadoutESN(seed=42, alpha_recon=1.0)
        cal = PercentileCalibrator(esn, max_history=16000)
        return SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)
    if arm == "P05":
        esn = PredictiveReadoutESN(seed=42, alpha_recon=0.5)
        cal = PercentileCalibrator(esn, max_history=16000)
        return SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)
    if arm == "zscore":
        return RollingRobustZ(window_size=50)
    if arm == "random":
        return RandomDetector(seed=42)
    raise ValueError(f"Brazo desconocido: {arm}")


def run_series_all_arms(file_path: Path, timeout_s: float = 3600.0) -> List[Dict[str, Any]]:
    """Procesa una serie para los 4 brazos garantizando ventana y recorte idénticos."""
    df = pd.read_csv(file_path).dropna()
    if df.empty:
        raise ValueError("Serie vacía")

    # Contrato verificado: 472 series son univariadas (col 0: datos, col 1: Label)
    data = df.iloc[:, 0].values.astype(float)
    labels = df["Label"].astype(int).to_numpy()
    n_points = len(data)

    # Familia extraída de la convención de nombre TSB-AD (p.ej. 029_WSD_id_1...)
    parts = file_path.name.split("_")
    family = parts[1] if len(parts) > 1 else "UNKNOWN"

    tsb_path = _resolve_tsb_path()
    if str(tsb_path) not in sys.path:
        sys.path.append(str(tsb_path))
    from TSB_AD.utils.slidingWindows import find_length_rank

    # slidingWindow idéntico calculado una sola vez
    sliding_window = find_length_rank(data.reshape(-1, 1), rank=1)

    results = []
    for arm in ARMS:
        det = factory(arm)
        scores = []
        t0 = time.time()
        failed = False
        for i, val in enumerate(data):
            if timeout_s is not None and (time.time() - t0) > timeout_s:
                failed = True
                break
            s = det.update(float(val), str(i))
            if math.isnan(s) or math.isinf(s):
                failed = True
                break
            scores.append(s)

        if failed or len(scores) != n_points:
            results.append({
                "series": file_path.name,
                "family": family,
                "n_points": n_points,
                "arm": arm,
                "auc_roc": np.nan,
                "auc_pr": np.nan,
                "vus_pr": np.nan,
            })
            continue

        # Recorte idéntico en [0, 1]
        clipped = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        m = _get_tsb_metrics(clipped, labels, sliding_window=sliding_window)
        results.append({
            "series": file_path.name,
            "family": family,
            "n_points": n_points,
            "arm": arm,
            "auc_roc": float(m.get("AUC-ROC", np.nan)),
            "auc_pr": float(m.get("AUC-PR", np.nan)),
            "vus_pr": float(m.get("VUS-PR", np.nan)),
        })

    return results


def append_rows_dict(csv_path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(csv_path)
    need_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if need_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
