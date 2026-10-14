"""Runner TSB-AD estrictamente causal (sin normalización batch global)."""

import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parents[3]


def _resolve_tsb_path() -> Path:
    candidates = [
        ROOT_DIR / "benchmarks" / "tsb_ad" / "TSB-AD",
        ROOT_DIR.parent / "benchmarks" / "tsb_ad" / "TSB-AD",
        Path(__file__).resolve().parents[4] / "benchmarks" / "tsb_ad" / "TSB-AD",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def get_tuning_files(list_file: str | os.PathLike) -> List[str]:
    df = pd.read_csv(list_file)
    return df[~df['file_name'].str.contains('_NAB_')]['file_name'].tolist()


def _causal_pred(scores: np.ndarray, warmup: int = 50) -> np.ndarray:
    """Umbral causal expansivo: pred[t] usa solo media/std de scores[:t].

    Sin fuga del futuro. Durante el warmup pred=0.
    """
    pred = np.zeros(len(scores), dtype=bool)
    if len(scores) == 0:
        return pred
    running_sum = 0.0
    running_sq = 0.0
    for t in range(len(scores)):
        if t < warmup:
            running_sum += float(scores[t])
            running_sq += float(scores[t]) ** 2
            continue
        n = t  # observaciones pasadas [0, t)
        mean = running_sum / max(n, 1)
        var = max(running_sq / max(n, 1) - mean ** 2, 0.0)
        std = math.sqrt(var)
        pred[t] = bool(scores[t] > (mean + 3.0 * std))
        running_sum += float(scores[t])
        running_sq += float(scores[t]) ** 2
    return pred


def _get_tsb_metrics(scores: np.ndarray, labels: np.ndarray, sliding_window: int) -> Dict[str, float]:
    tsb_path = _resolve_tsb_path()
    if str(tsb_path) not in sys.path:
        sys.path.append(str(tsb_path))

    from TSB_AD.evaluation.metrics import get_metrics

    # Sin MinMaxScaler global: los detectores ya emiten en [0, 1].
    # Solo recorte causal punto a punto, sin estadísticos del futuro.
    clipped = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    pred = _causal_pred(clipped)
    return get_metrics(clipped, labels, slidingWindow=sliding_window, pred=pred)


def run_single_series(detector_factory, file_path: str | os.PathLike,
                      timeout_s: float | None = 120.0,
                      channel: str = "first") -> Dict[str, Any]:
    """Ejecuta un detector causal sobre una serie TSB-AD.

    Args:
        detector_factory: fábrica ``() -> Detector`` con ``update(value, ts)``.
        file_path: CSV de la serie.
        timeout_s: límite por serie (None = sin límite).
        channel: ``"first"`` usa explícitamente la primera columna numérica
            (series multivariadas se reducen de forma documentada);
            ``"mean"`` promedia todas las columnas de datos por paso.
    """
    start_time = time.time()
    try:
        fp = Path(file_path)
        if not fp.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(fp).dropna()
        if df.empty:
            raise ValueError("Empty dataframe after dropna")

        data = df.iloc[:, 0:-1].values.astype(float)
        labels = df['Label'].astype(int).to_numpy()
        n_dim = int(data.shape[1])

        tsb_path = _resolve_tsb_path()
        if str(tsb_path) not in sys.path:
            sys.path.append(str(tsb_path))

        from TSB_AD.utils.slidingWindows import find_length_rank
        ref = data if n_dim == 1 else data[:, 0].reshape(-1, 1)
        sliding_window = find_length_rank(ref, rank=1)

        detector = detector_factory()
        scores = []
        t_det0 = time.time()
        for i in range(len(data)):
            if timeout_s is not None and time.time() - start_time > timeout_s:
                return {"status": "NO_EVALUADA", "error": "Timeout",
                        "time": time.time() - start_time, "length": 0,
                        "metrics": {}, "time_detector": 0.0, "time_metrics": 0.0}
            if channel == "mean" and n_dim > 1:
                val = float(np.mean(data[i, :]))
            else:
                # Reducción univariada explícita y documentada.
                val = float(data[i, 0])
            score = detector.update(val, str(i))
            if math.isnan(score) or math.isinf(score):
                return {"status": "NO_EVALUADA", "error": "Score is NaN/Inf",
                        "time": time.time() - start_time, "length": 0,
                        "metrics": {}, "time_detector": 0.0, "time_metrics": 0.0}
            scores.append(score)
        time_detector = time.time() - t_det0

        scores_arr = np.array(scores)
        t_met0 = time.time()
        metrics = _get_tsb_metrics(scores_arr, labels, sliding_window)
        time_metrics = time.time() - t_met0

        return {
            "status": "OK",
            "length": len(data),
            "n_dim": n_dim,
            "channel": channel,
            "metrics": metrics,
            "time": time.time() - start_time,
            "time_detector": time_detector,
            "time_metrics": time_metrics,
            "error": None,
        }
    except Exception as e:
        return {"status": "NO_EVALUADA", "error": type(e).__name__ + ": " + str(e),
                "time": time.time() - start_time, "length": 0,
                "metrics": {}, "time_detector": 0.0, "time_metrics": 0.0}
