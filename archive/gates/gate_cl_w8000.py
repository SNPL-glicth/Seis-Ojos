"""Gate reapertura: ContextualLikelihood W=8000 (paper Ahmad&Purdy 2016) — 1 config.

Unica diferencia vs el falsado §3.3: long_window 1000 -> 8000 (W'=10 igual).
Cableado IDENTICO al gate original (CL -> Cal -> Peak -> Spatial -> Refr) para
aislar la variable W. Dos variables a la vez seria inatribuible.

Predicciones firmadas ANTES de medir (2026-10-15):
  P1: FPs en Tweets <= 120 (baseline 214) al umbral optimo global.
  P2: >= 90 ventanas detectadas de 116 (baseline 97).
  P3: KnownCause no baja >1 ventana detectada vs baseline pareado.
  P4: media 3 semillas >= 60.68 + LOCO sin folds < -2.
Falsada si falla cualquiera. Baseline: sixeyes_abl_full_s42 (mismo metodo).

Uso: python archive/gates/gate_cl_w8000.py --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser as dp

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

from exp_replay_pareado import load_all
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.experimental.contextual_likelihood import ContextualLikelihood
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
BASELINE = "sixeyes_abl_full_s42"


class SeqPass:
    def __init__(self, seq):
        self.seq = [float(x) for x in seq]
        self.i = 0

    def update(self, v, t=""):
        r = self.seq[self.i]
        self.i += 1
        return r

    def reset(self):
        self.i = 0


def window_metrics(det_name):
    """FP-Tweets, ventanas totales y KnownCause al umbral optimo global del CSV."""
    df = pd.read_csv(NAB / "results" / det_name / f"{det_name}_standard_scores.csv")
    thr = float(df["Threshold"].mode()[0])
    windows = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    fp_tweets, hits, kc_hits, kc_tot = 0, 0, 0, 0
    for rel, wins in windows.items():
        cat, fn = rel.split("/", 1)
        d = pd.read_csv(NAB / "results" / det_name / cat / f"{det_name}_{fn}")
        s = d["anomaly_score"].values
        ts = pd.to_datetime(d["timestamp"])
        n = len(s)
        prob = min(int(0.15 * n), 750)
        fire = (s >= thr)
        fire[:prob] = False
        fp = int((fire & ~_inwin(ts, wins)).sum())
        if cat == "realTweets":
            fp_tweets += fp
        for a, b in wins:
            m = ((ts >= dp.parse(a)) & (ts <= dp.parse(b))).values
            idx = np.where(m)[0]
            idx = idx[idx >= prob]
            hit = bool(len(idx) and fire[idx].any())
            hits += hit
            if cat == "realKnownCause":
                kc_tot += 1
                kc_hits += hit
    return {"thr": thr, "fp_tweets": fp_tweets, "hits": hits, "kc_hits": kc_hits, "kc_tot": kc_tot}


def _inwin(ts, wins):
    m = np.zeros(len(ts), dtype=bool)
    for a, b in wins:
        m |= ((ts >= dp.parse(a)) & (ts <= dp.parse(b))).values
    return m


def run_gate(seed):
    col = f"s{seed}"
    det = f"sixeyes_clw8000_s{seed}"
    out = NAB / "results" / det
    out.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print(f"[gate_cl8000] seed={seed} series={len(data)}", flush=True)
    for rel, d in data.items():
        cl = ContextualLikelihood(short_window=10, long_window=8000)
        cl_errs = [cl.update(x) for x in d[col]]
        cal = PercentileCalibrator(SeqPass(cl_errs), max_history=16000)
        dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
        spat = SpatialSpikeBypass(dec, tolerance=0.05, warmup=100)
        refr = RefractorySuppressor(spat, w_rest=80, trigger_threshold=0.990)
        scores = [refr.update(v) for v in d["value"]]
        cat, fn = rel.split("/", 1)
        (out / cat).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": d["timestamp"], "value": d["value"],
                      "anomaly_score": scores,
                      "label": np.zeros(len(scores), dtype=int)}).to_csv(
            out / cat / f"{det}_{fn}", index=False)
    scores = NABRunner(nab_root=NAB).ejecutar_scorer_oficial(det)
    print(f"[scorer] {det}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{det}.json").write_text(json.dumps(
        {"detector": det, "seed": seed, "short": 10, "long": 8000,
         "scores": scores}, indent=2, sort_keys=True), encoding="utf-8")
    import shutil
    shutil.copy(out / f"{det}_standard_scores.csv", ROOT / "results" / f"{det}_standard_scores.csv")
    met = window_metrics(det)
    base = window_metrics(BASELINE)
    print(f"[metrics] {det}: {met} | baseline: {base}", flush=True)
    (ROOT / "results" / f"{det}_metrics.json").write_text(json.dumps(
        {"variant": met, "baseline": base}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, choices=[42, 7, 99])
    run_gate(ap.parse_args().seed)
