"""Gate oficial para ContextualLikelihood sobre ESN crudo (replay pareado).

Pipeline evaluado:
  ESN crudo -> ContextualLikelihood(short_window=10, long_window=1000)
            -> PercentileCalibrator(W=16000)
            -> StreamingPeakDecay(K=24, tau=2.0)
            -> SpatialSpikeBypass(tol=0.05, warmup=100)
            -> RefractorySuppressor(W=80, t=0.990)

Criterio pre-registrado estricto:
  1. Media Standard >= 60.68 (+3 sobre 57.68) en 3 semillas pareadas.
  2. Ninguna semilla peor vs titular pareado.
  3. LOCO-CV out-of-sample pareado no regresa.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
RAW = ROOT / "results" / "raw_esn"

K_WINDOW = 24
TAU = 2.0
CAL_W = 16000
SPATIAL_TOL = 0.05
SPATIAL_WARMUP = 100
REFR_W = 80
REFR_T = 0.990


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


def run_gate(seed: int, short_w: int = 10, long_w: int = 1000):
    col = f"s{seed}"
    det_name = f"sixeyes_cl_s{seed}"
    det_dir = NAB / "results" / det_name
    det_dir.mkdir(parents=True, exist_ok=True)

    series_dict = load_all()
    print(f"[gate_cl] semilla={seed} series={len(series_dict)} short={short_w} long={long_w}")

    for rel, data in series_dict.items():
        raws = data[col]
        vals = data["value"]
        ts = data["timestamp"]

        cl = ContextualLikelihood(short_window=short_w, long_window=long_w)
        cl_errs = [cl.update(x) for x in raws]

        cal = PercentileCalibrator(SeqPass(cl_errs), max_history=CAL_W)
        dec = StreamingPeakDecay(cal, k_window=K_WINDOW, tau=TAU)
        spat = SpatialSpikeBypass(dec, tolerance=SPATIAL_TOL, warmup=SPATIAL_WARMUP)
        refr = RefractorySuppressor(spat, w_rest=REFR_W, trigger_threshold=REFR_T)

        scores = [refr.update(v) for v in vals]

        cat, fn = rel.split("/", 1)
        out_sub = det_dir / cat
        out_sub.mkdir(parents=True, exist_ok=True)
        out_csv = out_sub / f"{det_name}_{fn}"
        df = pd.DataFrame({
            "timestamp": ts,
            "value": vals,
            "anomaly_score": scores,
            "raw_score": raws,
            "label": np.zeros(len(scores), dtype=int),
        })
        df.to_csv(out_csv, index=False)

    runner = NABRunner(nab_root=NAB)
    scores = runner.ejecutar_scorer_oficial(det_name)

    res_json = ROOT / "results" / f"{det_name}.json"
    res_json.write_text(json.dumps({
        "detector": det_name,
        "seed": seed,
        "short_window": short_w,
        "long_window": long_w,
        "scores": scores
    }, indent=2), encoding="utf-8")

    import shutil
    shutil.copy(det_dir / f"{det_name}_standard_scores.csv",
                ROOT / "results" / f"{det_name}_standard_scores.csv")

    print(f"\n[gate_cl] RESULTADOS OFICIALES NAB {det_name}:")
    for k, v in scores.items():
        print(f"   {k:25s} : {v:.2f}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--short", type=int, default=10)
    parser.add_argument("--long", type=int, default=1000)
    args = parser.parse_args()
    run_gate(seed=args.seed, short_w=args.short, long_w=args.long)
