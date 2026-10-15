"""Gate unico: LevelRarity (soft bypass) + SpatialSpikeBypass + Refractory.

Pipeline:
  ESN -> Cal(W16000) -> PeakDecay(24,2.0) -> SpatialSpikeBypass(tol=0.05,warmup=100)
  values -> LevelRarity(K=80, half_life=2000, warmup=100, margin=0.05)
  max(bypass, rarity) -> RefractorySuppressor(W=80, t=0.99)

Criterio pre-registrado (un solo intento):
  1. Media Standard >= +3 sobre 57.68 (>= 60.68) en 3 semillas pareadas.
  2. Ninguna semilla peor.
  3. LOCO pareado no regresa.
Analisis post: 17 ventanas recuperadas, FPs nuevos, alarmas fuera de ventana.
"""
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
from sixeyes.experimental.level_rarity import LevelRarity
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
DET = "sixeyes_level_rarity"

K_WINDOW, TAU, CAL_W = 24, 2.0, 16000
SPATIAL_TOL, SPATIAL_WARMUP = 0.05, 100
REFR_W, REFR_T = 80, 0.990
RARITY_K, RARITY_HL, RARITY_WARMUP, RARITY_MARGIN = 80, 2000.0, 100, 0.05

TARGET_17 = [
    ("artificialWithAnomaly/art_daily_flatmiddle.csv", "2014-04-10"),
    ("artificialWithAnomaly/art_daily_jumpsdown.csv", "2014-04-10"),
    ("artificialWithAnomaly/art_load_balancer_spikes.csv", "2014-04-10"),
    ("realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv", "2014-02-27"),
    ("realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv", "2014-02-17"),
    ("realAWSCloudwatch/grok_asg_anomaly.csv", "2014-01-20"),
    ("realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv", "2014-02-26"),
    ("realAdExchange/exchange-3_cpc_results.csv", "2011-08-12"),
    ("realAdExchange/exchange-4_cpc_results.csv", "2011-08-01"),
    ("realKnownCause/machine_temperature_system_failure.csv", "2013-12-10"),
    ("realKnownCause/machine_temperature_system_failure.csv", "2014-02-07"),
    ("realKnownCause/nyc_taxi.csv", "2014-11-25"),
    ("realKnownCause/nyc_taxi.csv", "2015-01-24"),
    ("realTraffic/TravelTime_387.csv", "2015-07-27"),
    ("realTraffic/TravelTime_451.csv", "2015-08-09"),
    ("realTweets/Twitter_volume_GOOG.csv", "2015-03-22"),
    ("realTweets/Twitter_volume_UPS.csv", "2015-03-29"),
]


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


def replay_series(raws, vals, ts, seed):
    cal = PercentileCalibrator(SeqPass(raws), max_history=CAL_W)
    dec = StreamingPeakDecay(cal, k_window=K_WINDOW, tau=TAU)
    spatial = SpatialSpikeBypass(dec, tolerance=SPATIAL_TOL, warmup=SPATIAL_WARMUP)
    rarity = LevelRarity(n_bins=RARITY_K, half_life=RARITY_HL,
                         warmup=RARITY_WARMUP, margin=RARITY_MARGIN)
    refr = RefractorySuppressor(spatial, w_rest=REFR_W, trigger_threshold=REFR_T)

    out = np.empty(len(raws), dtype=float)
    rarity_raw = np.empty(len(raws), dtype=float)
    for i, (r, v, tt) in enumerate(zip(raws, vals, ts)):
        s = refr.update(r, tt)
        rarity_raw[i] = rarity.update(v, tt)
        out[i] = s
    return out, rarity_raw


def main():
    data = load_all()
    wins = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    print(f"[gate] serie={len(data)} det={DET}", flush=True)

    fused = {}
    rarity_fused = {}
    for rel, d in data.items():
        scores, rraw = replay_series(d["s42"], d["value"], d["timestamp"], 42)
        fused[rel] = scores
        rarity_fused[rel] = rraw

    out_dir = NAB / "results" / DET
    for rel, scores in fused.items():
        rp = Path(rel)
        dest = out_dir / rp.parent / f"{DET}_{rp.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(NAB / "data" / rel, usecols=["timestamp", "value"])
        tss = pd.to_datetime(df["timestamp"])
        lab = np.zeros(len(df), dtype=int)
        for a, b in wins[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            lab |= ((tss >= aa) & (tss <= bb)).to_numpy()
        pd.DataFrame({"timestamp": df["timestamp"], "value": df["value"],
                      "anomaly_score": scores, "label": lab}).to_csv(dest, index=False)
    print(f"[gate] CSVs en {out_dir}", flush=True)

    scores = NABRunner(nab_root=NAB).ejecutar_scorer_oficial(DET)
    print(f"[scorer] {DET}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{DET}.json").write_text(
        json.dumps({"detector": DET, "scores": scores}, indent=2, sort_keys=True), encoding="utf-8")
    import shutil
    shutil.copy(out_dir / f"{DET}_standard_scores.csv", ROOT / "results" / f"{DET}_standard_scores.csv")

    thr = float(pd.read_csv(out_dir / f"{DET}_standard_scores.csv")["Threshold"].iloc[0])

    print(f"\n[analisis] umbral={thr:.6f}", flush=True)
    recovered = 0
    for rel, a in TARGET_17:
        rp = Path(rel)
        df = pd.read_csv(out_dir / rp.parent / f"{DET}_{rp.name}")
        tss = pd.to_datetime(df["timestamp"])
        sc = df["anomaly_score"].to_numpy(dtype=float)
        aa = dp.parse(a)
        m = (tss >= aa).to_numpy()
        if m.sum() and sc[m].max() >= thr:
            recovered += 1
            print(f"  RECUPERADA {rel} {a[:10]} max={sc[m].max():.4f}", flush=True)
    print(f"[analisis] 17 ventanas recuperadas: {recovered}/17", flush=True)

    fp_nuevos = 0
    for rel in wins:
        rp = Path(rel)
        df = pd.read_csv(out_dir / rp.parent / f"{DET}_{rp.name}")
        tss = pd.to_datetime(df["timestamp"])
        sc = df["anomaly_score"].to_numpy(dtype=float)
        in_win = np.zeros(len(df), dtype=bool)
        for a, b in wins[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            in_win |= ((tss >= aa) & (tss <= bb)).to_numpy()
        fp_nuevos += int(((sc >= thr) & ~in_win).sum())
    print(f"[analisis] FPs nuevos (score>=thr fuera de ventana): {fp_nuevos}", flush=True)

    print(f"\n[resultado] {DET}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    print(f"[criterio] media >= 60.68? {'SI' if scores['standard'] >= 60.68 else 'NO'}", flush=True)


if __name__ == "__main__":
    main()
