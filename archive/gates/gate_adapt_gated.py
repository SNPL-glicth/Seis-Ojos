"""Gate 1-intento: AdaptiveThreshold + GatedRefractory sobre raw_esn (3 semillas).

Replay offline bit-equivalente al pipeline online (misma composicion de
clases reales, sin re-ejecutar el ESN: se inyectan los crudos s42/s7/s99):
  SeqPass(raw) -> Cal(W16000) -> PeakDecay(24,2.0) -> CausalAdaptiveThreshold
  -> SpatialSpikeBypass(0.05/100) -> RegimeGatedRefractory
El bypass opera sobre el valor real x_t (igual que online); el resto sobre
el flujo de scores. Escribe NAB/results/<det>/ (58 CSVs) + scorer oficial.

Uso:
  python archive/gates/gate_adapt_gated.py --seed 42
Criterio: media 3 semillas >= 60.68, ninguna peor que su full pareado.
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

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.experimental.causal_adaptive_threshold import CausalAdaptiveThreshold
from sixeyes.experimental.regime_gated_refractory import RegimeGatedRefractory
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
RAW = ROOT / "results" / "raw_esn"


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


def build_chain(raw_seq):
    cal = PercentileCalibrator(SeqPass(raw_seq), max_history=16000)
    dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    adapt = CausalAdaptiveThreshold(dec)
    spat = SpatialSpikeBypass(adapt, tolerance=0.05, warmup=100)
    gated = RegimeGatedRefractory(spat)
    return gated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, choices=[42, 7, 99])
    args = ap.parse_args()
    seedkey = f"s{args.seed}"
    det = f"sixeyes_adapt_gated_s{args.seed}"

    windows = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    files = sorted(RAW.glob("*.npz"))
    print(f"[gate] adapt_gated seed={args.seed} series={len(files)}", flush=True)
    fused = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        rel = f.stem.replace("__", "/")
        chain = build_chain(np.array(z[seedkey], dtype=float))
        vals = np.array(z["value"], dtype=float)
        ts = [str(x) for x in z["timestamp"].tolist()]
        out = np.empty(len(vals), dtype=float)
        for i, (v, tt) in enumerate(zip(vals, ts)):
            out[i] = chain.update(float(v), tt)
        fused[rel] = out

    out_dir = NAB / "results" / det
    for rel, scores in fused.items():
        rp = Path(rel)
        dest = out_dir / rp.parent / f"{det}_{rp.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp", "value"])
        tss = pd.to_datetime(df["timestamp"])
        lab = np.zeros(len(df), dtype=int)
        for a, b in windows[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            lab |= ((tss >= aa) & (tss <= bb)).to_numpy()
        pd.DataFrame({"timestamp": df["timestamp"], "value": df["value"],
                      "anomaly_score": scores, "label": lab}).to_csv(dest, index=False)
    print(f"[gate] CSVs en {out_dir}", flush=True)
    runner = NABRunner(nab_root=NAB)
    scores = runner.ejecutar_scorer_oficial(det)
    print(f"[scorer] {det}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{det}.json").write_text(
        json.dumps({"detector": det, "seed": seedkey, "scores": scores},
                   indent=2, sort_keys=True), encoding="utf-8")
    import shutil
    shutil.copy(out_dir / f"{det}_standard_scores.csv",
                ROOT / "results" / f"{det}_standard_scores.csv")


if __name__ == "__main__":
    main()
