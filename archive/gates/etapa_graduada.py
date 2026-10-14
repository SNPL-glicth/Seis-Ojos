"""Etapa de decisión GRADUADA sobre k-NN guardado (un solo intento).

Igual que la binaria (umbral adaptativo μ+3σ trailing W=500, enmascarado
100) pero emisión continua: (s−thr)/(1−thr) recortado a [0,1], 0 fuera.
El optimizador recupera su herramienta recall↔FP (en binaria fijaba 1.0).
Regla: suma ≥+3 NAB combinado con la atribución (que sumó +0) o se cierra.
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from dateutil import parser as dp

ROOT = Path(__file__).resolve().parents[2]
NAB = (ROOT.parent / "NAB").resolve()

W = 500
K_SIGMA = 3.0
MASK = 100


def emitir(scores: np.ndarray) -> np.ndarray:
    s = pd.Series(scores)
    mu = s.rolling(W, min_periods=50).mean().to_numpy()
    sd = s.rolling(W, min_periods=50).std(ddof=0).to_numpy()
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where(np.isnan(sd) | (sd <= 0), 1e-12, sd)
    thr = mu + K_SIGMA * sd
    out = np.zeros(len(s))
    masked_until = -1
    for i in range(len(s)):
        if i <= masked_until:
            continue
        if s[i] > thr[i]:
            denom = max(1.0 - thr[i], 1e-12)
            out[i] = min(max((s[i] - thr[i]) / denom, 0.0), 1.0)
            masked_until = i + MASK
    return out


def main():
    from sixeyes.evaluation.nab_runner import NABRunner  # noqa
    sys.path.insert(0, str(ROOT / "src"))
    labels = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    base = "sixeyes_knn_retardos_bA_k12_tau2p0_w16000"
    out = "sixeyes_decision_graduada_knn"
    src = NAB / "results" / base
    dst = NAB / "results" / out
    for rel in labels:
        rp = Path(rel)
        df = pd.read_csv(src / rp.parent / f"{base}_{rp.name}",
                         usecols=["timestamp", "value", "anomaly_score"])
        em = emitir(df["anomaly_score"].to_numpy())
        ts = pd.to_datetime(df["timestamp"])
        lab = np.zeros(len(df), dtype=int)
        for a, b in labels[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            lab |= ((ts >= aa) & (ts <= bb)).to_numpy()
        dest = dst / rp.parent / f"{out}_{rp.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": df["timestamp"], "value": df["value"],
                      "anomaly_score": em, "label": lab}).to_csv(dest, index=False)
    print("emisiones graduadas listas", flush=True)
    scores = NABRunner(nab_root=NAB).ejecutar_scorer_oficial(out)
    (ROOT / "results" / "sixeyes_decision_graduada_knn.json").write_text(
        json.dumps({"scores": scores}, indent=2, sort_keys=True))
    print(scores, flush=True)


if __name__ == "__main__":
    main()
