"""Comparacion gratuita: relativeEntropy vs spatial-s42 en las 17 ventanas.

relativeEntropy (Wang et al. 2011): histogramas de 5 bins en ventanas de 52,
memoria de hipotesis, nueva categoria si ninguna encaja. Casi el concepto
ART-sin-neuronas. NAB oficial: 54.64.
Pregunta: cuantas de las 17 ventanas de la brecha cubre, y con cuantos FPs?
Si >= 8-10 con pocos FPs, merece construir version propia. Si pocas, archivar.
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

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")

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


def load(det):
    det_dir = NAB / "results" / det
    thr = float(pd.read_csv(det_dir / f"{det}_standard_scores.csv")["Threshold"].iloc[0])
    frames = {}
    for csv in det_dir.rglob(f"{det}_*.csv"):
        if csv.name.endswith("_standard_scores.csv") or "reward" in csv.name:
            continue
        rel = f"{csv.parent.name}/{csv.name[len(det) + 1:-4]}.csv"
        frames[rel] = pd.read_csv(csv, usecols=["timestamp", "anomaly_score"])
    return frames, thr


def main():
    wins = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    fr_re, thr_re = load("relativeEntropy")
    fr_sp, thr_sp = load("sixeyes_abl_spatial_s42")
    print(f"umbral relativeEntropy={thr_re:.5f} spatial={thr_sp:.5f}", flush=True)

    cubiertas = 0
    for rel, a in TARGET_17:
        key = next((k for k in fr_re if k == rel), None)
        if not key:
            print(f"  SIN DATOS {rel}", flush=True)
            continue
        df = fr_re[key]
        tss = pd.to_datetime(df["timestamp"])
        sc = df["anomaly_score"].to_numpy(dtype=float)
        aa = dp.parse(a)
        m = (tss >= aa).to_numpy()
        mx = sc[m].max() if m.sum() else 0.0
        hit = mx >= thr_re
        cubiertas += int(hit)
        print(f"  {'SI ' if hit else 'no '} {rel:55s} {a[:10]} max_re={mx:.4f}", flush=True)
    print(f"\nrelativeEntropy cubre {cubiertas}/17 de la brecha", flush=True)

    fp = 0
    for rel in wins:
        key = next((k for k in fr_re if k == rel), None)
        if not key:
            continue
        df = fr_re[key]
        tss = pd.to_datetime(df["timestamp"])
        sc = df["anomaly_score"].to_numpy(dtype=float)
        in_win = np.zeros(len(df), dtype=bool)
        for a, b in wins[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            in_win |= ((tss >= aa) & (tss <= bb)).to_numpy()
        fp += int(((sc >= thr_re) & ~in_win).sum())
    print(f"FPs de relativeEntropy (thr propio): {fp}", flush=True)
    print(f"\nveredicto: {'CONSTRUIR version propia' if cubiertas >= 8 else 'archivar concepto'}", flush=True)


if __name__ == "__main__":
    main()
