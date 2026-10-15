"""Diagnostico de los 418 FPs de LevelRarity: agrupamiento y permanencia.

Sin corridas nuevas: lee los CSVs por punto de sixeyes_level_rarity y el umbral
optimo, agrupa FPs en episodios (gap <= 80, el silencio del refractario), y mide
que fraccion es permanencia (puntos tras el inicio del episodio) vs entrada.
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

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
DET = "sixeyes_level_rarity"

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

GAP = 80


def main():
    wins = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    thr = float(pd.read_csv(NAB / "results" / DET / f"{DET}_standard_scores.csv")["Threshold"].iloc[0])
    print(f"umbral={thr:.6f}", flush=True)

    total_fp = 0
    episodios = []
    permanencia_pts = 0
    entrada_pts = 0
    tp_inicio = 0
    tp_total = 0

    for rel in wins:
        rp = Path(rel)
        csv = NAB / "results" / DET / rp.parent / f"{DET}_{rp.name}"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        tss = pd.to_datetime(df["timestamp"])
        sc = df["anomaly_score"].to_numpy(dtype=float)
        in_win = np.zeros(len(df), dtype=bool)
        for a, b in wins[rel]:
            aa, bb = dp.parse(a), dp.parse(b)
            in_win |= ((tss >= aa) & (tss <= bb)).to_numpy()

        alerta = sc >= thr
        fp_idx = np.where(alerta & ~in_win)[0]
        tp_idx = np.where(alerta & in_win)[0]

        targets = [(t[1], t[1]) for t in TARGET_17 if t[0] == rel]
        hit_tp = set()
        for a, b in targets:
            aa, bb = dp.parse(a), dp.parse(b)
            m = ((tss >= aa) & (tss <= bb)).to_numpy()
            if m.sum() and sc[m].max() >= thr:
                hit_tp.update(np.where(m)[0].tolist())
        tp_total += len(targets)
        tp_inicio += len(hit_tp)

        if len(fp_idx) == 0:
            continue

        ep = [[fp_idx[0]]]
        for i in fp_idx[1:]:
            if i - ep[-1][-1] <= GAP:
                ep[-1].append(i)
            else:
                ep.append([i])

        for e in ep:
            total_fp += len(e)
            episodios.append((rel, len(e), e[0], e[-1]))
            inicio = e[0]
            for j in e:
                if j - inicio > GAP:
                    permanencia_pts += 1
                else:
                    entrada_pts += 1

    print(f"\ntotal FPs: {total_fp}", flush=True)
    print(f"episodios distintos (gap<={GAP}): {len(episodios)}", flush=True)
    print(f"  FPs en entrada (primeros {GAP} pasos del episodio): {entrada_pts}", flush=True)
    print(f"  FPs en permanencia (tras {GAP} pasos): {permanencia_pts}", flush=True)
    if total_fp:
        print(f"  fraccion permanencia: {permanencia_pts/total_fp:.2%}", flush=True)

    lens = sorted((e[1] for e in episodios), reverse=True)
    print(f"\nduracion de episodios: max={lens[0]}, mediana={np.median(lens):.0f}", flush=True)
    print(f"top 10 duraciones: {lens[:10]}", flush=True)

    n_1 = sum(1 for l in lens if l == 1)
    n_80 = sum(1 for l in lens if l >= 80)
    print(f"episodios de 1 punto: {n_1}", flush=True)
    print(f"episodios >=80 puntos: {n_80}", flush=True)


if __name__ == "__main__":
    main()
