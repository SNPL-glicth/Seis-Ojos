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

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Set

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


def assert_frozen(tag: str = "prereg-p05-v1") -> str:
    """Verifica que el estado actual del repo coincida exactamente con el commit del tag y sin cambios dirty."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tagc = subprocess.check_output(["git", "rev-parse", f"{tag}^{{commit}}"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    assert head == tagc, f"HEAD {head} != {tag} {tagc}"
    assert not dirty, f"Hay cambios sin commit en archivos versionados:\n{dirty}"
    return head


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


def run_series_all_arms(file_path_str: str, timeout_s: float = 3600.0) -> List[Dict[str, Any]]:
    """Procesa una serie para los 4 brazos garantizando ventana y recorte idénticos."""
    file_path = Path(file_path_str)
    df = pd.read_csv(file_path).dropna()
    if df.empty:
        raise ValueError(f"Serie vacía: {file_path}")

    # Contrato verificado: 472 series son univariadas (col 0: datos, col 1: Label)
    data = df.iloc[:, 0].values.astype(float)
    labels = df["Label"].astype(int).to_numpy()
    n_points = len(data)

    # Familia extraída de convención TSB-AD (p.ej. 029_WSD_id_1...)
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


def get_fresh_series() -> List[str]:
    full = set(pd.read_csv(FL / "TSB-AD-U-Eva-Full.csv")["file_name"])
    eva = set(pd.read_csv(FL / "TSB-AD-U-Eva.csv")["file_name"])
    fresh = sorted(full - eva)
    assert len(fresh) == 472, f"Se esperaban 472 frescas, encontradas {len(fresh)}"
    return fresh


def get_completed_series(csv_path: Path) -> Set[str]:
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path)
    # Una serie está completa si tiene los 4 brazos registrados
    counts = df.groupby("series")["arm"].nunique()
    return set(counts[counts >= len(ARMS)].index)


def main():
    parser = argparse.ArgumentParser(description="Runner congelado de las 472 frescas.")
    parser.add_argument("--out-csv", default="results/frescas_frozen_p05.csv")
    parser.add_argument("--meta-json", default="results/frescas_frozen_meta.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    # Guardián del protocolo: assert_frozen
    commit_hash = assert_frozen(tag="prereg-p05-v1")
    print(f"[GUARDIÁN ACTIVO] Protocolo verificado contra commit {commit_hash}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    meta_json = Path(args.meta_json)
    with open(meta_json, "w", encoding="utf-8") as mf:
        json.dump({
            "tag": "prereg-p05-v1",
            "commit_hash": commit_hash,
            "start_time": time.time(),
            "arms": ARMS,
            "fieldnames": FIELDNAMES,
        }, mf, indent=2)

    fresh = get_fresh_series()
    done = get_completed_series(out_csv)
    pending = [f for f in fresh if f not in done]
    print(f"[RUNNER] Total frescas: {len(fresh)} | Completadas: {len(done)} | Pendientes: {len(pending)}")

    if not pending:
        print("[RUNNER] Todas las series ya están completadas.")
        return

    t_start = time.time()
    n_total = len(pending)
    completed_now = 0

    if args.workers <= 1:
        for f in pending:
            res = run_series_all_arms(str(TSBU / f), timeout_s=args.timeout)
            append_rows_dict(out_csv, res)
            completed_now += 1
            el = time.time() - t_start
            print(f"[{completed_now}/{n_total}] {f} completada | {el:.1f}s acum", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(run_series_all_arms, str(TSBU / f), args.timeout): f
                for f in pending
            }
            for future in as_completed(future_to_file):
                f = future_to_file[future]
                try:
                    res = future.result()
                    append_rows_dict(out_csv, res)
                    completed_now += 1
                    el = time.time() - t_start
                    pace = el / completed_now
                    rem = pace * (n_total - completed_now)
                    print(f"[{completed_now}/{n_total}] {f} completada | {el:.0f}s acum (restante est: {rem:.0f}s)", flush=True)
                except Exception as e:
                    print(f"[ERROR] Serie {f} falló: {e}", flush=True)

    print(f"[RUNNER] Corrida finalizada en {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
