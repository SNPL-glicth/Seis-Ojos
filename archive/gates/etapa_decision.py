"""Etapa de decisión sobre scores guardados (diseño propio, sin detectores nuevos).

Emisión dispersa 0/1 cuando el score supera media+3σ trailing (W=500) de la
propia serie + enmascarado de 100 pts tras cada emisión. Constantes a priori.
Regla: gana con ≥+5 NAB o +6 ventanas netas sobre la base (con conteo
ganadas/perdidas); si no, se abandona la línea.
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from dateutil import parser as dp

ROOT = Path(__file__).resolve().parents[2]
NAB = (ROOT.parent / "NAB").resolve()
sys.path.insert(0, str(NAB))

W = 500
K_SIGMA = 3.0
MASK = 100


def emitir(scores: np.ndarray) -> np.ndarray:
    s = pd.Series(scores)
    mu = s.rolling(W, min_periods=50).mean().to_numpy()
    sd = s.rolling(W, min_periods=50).std(ddof=0).to_numpy()
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where(np.isnan(sd) | (sd <= 0), 1e-12, sd)
    out = np.zeros(len(s))
    masked_until = -1
    for i in range(len(s)):
        if i <= masked_until:
            continue
        if s[i] > mu[i] + K_SIGMA * sd[i]:
            out[i] = 1.0
            masked_until = i + MASK
    return out


def procesar(det_base: str, det_out: str) -> None:
    labels = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    src = NAB / "results" / det_base
    dst = NAB / "results" / det_out
    for rel in labels:
        rp = Path(rel)
        df = pd.read_csv(src / rp.parent / f"{det_base}_{rp.name}",
                         usecols=["timestamp", "value", "anomaly_score"])
        em = emitir(df["anomaly_score"].to_numpy())
        ts = pd.to_datetime(df["timestamp"])
        lab = np.zeros(len(df), dtype=int)
        for a, b in labels[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            lab |= ((ts >= aa) & (ts <= bb)).to_numpy()
        dest = dst / rp.parent / f"{det_out}_{rp.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": df["timestamp"], "value": df["value"],
                      "anomaly_score": em, "label": lab}).to_csv(dest, index=False)
    print(f"emisiones listas: {det_out}", flush=True)


def ventanas(det: str, thr: float) -> dict:
    labels = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    out = {}
    for rel, wins in labels.items():
        rp = Path(rel)
        df = pd.read_csv(NAB / "results" / det / rp.parent / f"{det}_{rp.name}",
                         usecols=["timestamp", "anomaly_score"])
        ts = pd.to_datetime(df["timestamp"])
        sc = df["anomaly_score"].to_numpy()
        for j, (a, b) in enumerate(wins):
            aa, bb = dp.parse(a), dp.parse(b)
            idx = df.index[(ts >= aa) & (ts <= bb)].to_numpy()
            out[(rel, j)] = bool(len(idx) and (sc[idx] >= thr).any())
    return out


def main():
    from sixeyes.evaluation.nab_runner import NABRunner  # noqa
    sys.path.insert(0, str(ROOT / "src"))
    base = "sixeyes_knn_retardos_bA_k12_tau2p0_w16000"
    out = "sixeyes_decision_knn"
    procesar(base, out)
    runner = NABRunner(nab_root=NAB)
    scores = runner.ejecutar_scorer_oficial(out)
    (ROOT / "results" / "sixeyes_decision_knn.json").write_text(
        json.dumps({"scores": scores}, indent=2, sort_keys=True))
    print(scores, flush=True)
    th = json.load(open(NAB / "config" / "thresholds.json", encoding="utf-8"))
    thr_base = th[base]["standard"]["threshold"]
    thr_new = th[out]["standard"]["threshold"]
    wb = ventanas(base, thr_base)
    wn = ventanas(out, thr_new)
    keys = set(wb) | set(wn)
    gan = sum(1 for k in keys if wn.get(k) and not wb.get(k))
    per = sum(1 for k in keys if wb.get(k) and not wn.get(k))
    print(f"ventanas: base={sum(wb.values())} nueva={sum(wn.values())} "
          f"ganadas={gan} perdidas={per} neto={gan - per}", flush=True)


if __name__ == "__main__":
    main()
