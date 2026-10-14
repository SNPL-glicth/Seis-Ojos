"""Métrica de desarrollo: AUC por ventana contra nulo + bootstrap por archivo.

Para cada ventana: rank del máximo real entre 200 nulas de igual longitud
en la misma serie (≈AUC). Agregado por archivo y media global con IC
bootstrap (1000 remuestreos de archivos). Sin umbral ⇒ menos ruido que NAB.
Uso: python archive/gates/metrica_auc_ventana.py <detector_dir> [solo_perdidas]
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from dateutil import parser as dp

ROOT = Path(__file__).resolve().parents[2]
NAB = (ROOT.parent / "NAB").resolve()

rng = np.random.RandomState(0)
NULL_REPS = 200
EARLY_FRAC = 0.30


def _ventana_max(sc, idx, frac=1.0):
    n = max(1, int(len(idx) * frac))
    return sc[idx[:n]].max()


def auc_ventana(det_dir: str, solo_perdidas: bool = False, thr: float = 0.99658055,
                early_frac: float = 1.0):
    labels = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    por_archivo: dict[str, list[float]] = {}
    for rel, wins in labels.items():
        rp = Path(rel)
        df = pd.read_csv(NAB / "results" / det_dir / rp.parent / f"{det_dir}_{rp.name}",
                         usecols=["timestamp", "anomaly_score"])
        ts = pd.to_datetime(df["timestamp"])
        sc = df["anomaly_score"].to_numpy()
        for a, b in wins:
            aa, bb = dp.parse(a), dp.parse(b)
            idx = df.index[(ts >= aa) & (ts <= bb)].to_numpy()
            if len(idx) == 0:
                continue
            real_max = _ventana_max(sc, idx, early_frac)
            if solo_perdidas and sc[idx].max() >= thr:
                continue
            L = max(1, int(len(idx) * early_frac))
            if len(sc) - L <= 0:
                continue
            null_maxes = np.array([sc[s:s + L].max()
                                   for s in rng.randint(0, len(sc) - L, size=NULL_REPS)])
            por_archivo.setdefault(rel, []).append(float(np.mean(null_maxes <= real_max)))
    medias = {r: float(np.mean(v)) for r, v in por_archivo.items()}
    return medias


def _ic(medias: dict[str, float]):
    arr = np.array(list(medias.values()))
    boot = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(1000)]
    return float(arr.mean()), (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def diff_pareada(det_a: str, det_b: str, early_frac: float = 1.0):
    """Bootstrap pareado de la diferencia (mismos índices): IC angosto."""
    ma = auc_ventana(det_a, early_frac=early_frac)
    mb = auc_ventana(det_b, early_frac=early_frac)
    rels = sorted(set(ma) & set(mb))
    da = np.array([ma[r] for r in rels])
    db = np.array([mb[r] for r in rels])
    diff = float(np.mean(db - da))
    boot = [float(np.mean((db - da)[rng.choice(len(rels), size=len(rels), replace=True)]))
            for _ in range(2000)]
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    print(f"pareada {det_a[:28]} vs {det_b[:28]} (frac={early_frac}): "
          f"diff={diff:+.4f} IC95=[{lo:+.4f},{hi:+.4f}] n={len(rels)}")
    return diff, (lo, hi)


if __name__ == "__main__":
    det = sys.argv[1] if len(sys.argv) > 1 else "sixeyes_esn_k24_tau2p0_w16000"
    if len(sys.argv) > 3 and sys.argv[2] == "vs":
        diff_pareada(det, sys.argv[3],
                     early_frac=float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
    else:
        perdidas = len(sys.argv) > 2 and sys.argv[2] == "solo_perdidas"
        m, (lo, hi) = _ic(auc_ventana(det, perdidas))
        print(f"{det} solo_perdidas={perdidas}: AUC={m:.3f} IC95=[{lo:.3f},{hi:.3f}]")
