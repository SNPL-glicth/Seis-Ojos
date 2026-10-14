"""Controles al diagnóstico señal-vs-umbral (sin detectores nuevos).

A. Control nulo del percentil: cada ventana perdida se rankea contra 200
   ventanas aleatorias de igual longitud en la misma serie.
B. Cota oráculo en detector débil (doble escala 45.19): mismo barrido
   por-archivo; si también sube ~+25, la cota no mide capacidad.
C. No-op del reescalado: Spearman(score, score/q99.9 trailing W=2000/16000).
   ~1 ⇒ no hay experimento que hacer.
D. Predictibilidad del umbral óptimo: correlación cruzada (Spearman +
   R² leave-one-out lineal) entre umbral óptimo por archivo y propiedades
   sin etiquetas (largo, curtosis del error crudo, fracción de empates en
   scores, agrupamiento de alarmas).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser as dp
from scipy import stats

sys.path.insert(0, "src")

NAB = Path(__file__).resolve().parents[2].parent / "NAB"
ESN = "sixeyes_esn_k24_tau2p0_w16000"
DUAL = "sixeyes_dualscale_b15_k12_tau2p0_w16000"
sys.path.insert(0, str(NAB))
from nab.sweeper import Sweeper  # type: ignore

rng = np.random.RandomState(0)


def load_det(det_dir, rel):
    rp = Path(rel)
    df = pd.read_csv(NAB / "results" / det_dir / rp.parent / f"{det_dir}_{rp.name}",
                     usecols=["timestamp", "anomaly_score"])
    return pd.to_datetime(df["timestamp"]), df["anomaly_score"].to_numpy()


def main():
    profiles = json.load(open(NAB / "config" / "profiles.json", encoding="utf-8"))
    labels = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    cm = profiles["standard"]["CostMatrix"]
    sw = Sweeper(probationPercent=0.15, costMatrix=cm)

    # A. control nulo (ventanas perdidas del ESN, umbral oficial 0.99658)
    THR = 0.99658055
    ranks = []
    for rel, wins in labels.items():
        ts, sc = load_det(ESN, rel)
        t = pd.Series(ts)
        for a, b in wins:
            aa, bb = dp.parse(a), dp.parse(b)
            idx = np.where(((t >= aa) & (t <= bb)).to_numpy())[0]
            if len(idx) == 0 or sc[idx].max() >= THR:
                continue
            real_max = sc[idx].max()
            L = len(idx)
            null_maxes = []
            for _ in range(200):
                s = rng.randint(0, len(sc) - L)
                null_maxes.append(sc[s:s + L].max())
            ranks.append(float(np.mean(np.array(null_maxes) <= real_max)))
    ranks = np.array(ranks)
    print(f"(A) n={len(ranks)} med={np.median(ranks):.3f} "
          f"frac>0.95={np.mean(ranks > 0.95):.2f} frac>0.99={np.mean(ranks > 0.99):.2f}")

    # B. cota oráculo en débil (dual 45.19)
    for det, tag in ((ESN, "esn"), (DUAL, "dual")):
        tot_best, tot_glob, nthr = 0.0, 0.0, []
        for rel, wins in labels.items():
            ts, sc = load_det(det, rel)
            wl = [(dp.parse(a), dp.parse(b)) for a, b in wins]
            alist = sw.calcSweepScore(list(ts), list(sc), list(wl), rel)
            sweep = sw.calcScoreByThreshold(alist)
            best = max(sweep, key=lambda x: x.score)
            tot_best += best.score
            nthr.append(best.threshold)
            _, mrow = sw.scoreDataSet(ts, sc, wl, rel, THR if tag == "esn" else 0.99781022)
            tot_glob += mrow.score if mrow is not None else 0.0
        print(f"(B) {tag}: global={tot_glob:.2f} por-archivo={tot_best:.2f} "
              f"lift={tot_best - tot_glob:.2f}")

    # C. no-op del reescalado (serie IBM como sonda + 5 más)
    probe = ["realTweets/Twitter_volume_IBM.csv",
             "realAdExchange/exchange-3_cpc_results.csv",
             "realTraffic/speed_7578.csv",
             "realKnownCause/nyc_taxi.csv",
             "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
             "artificialWithAnomaly/art_daily_jumpsup.csv"]
    for rel in probe:
        _, sc = load_det(ESN, rel)
        s = pd.Series(sc)
        for W in (2000, 16000):
            q = s.rolling(W, min_periods=50).quantile(0.999).to_numpy()
            q = np.where((q == 0) | np.isnan(q), 1.0, q)
            rho, _ = stats.spearmanr(sc, sc / q)
            print(f"(C) {rel.split('/')[-1][:24]} W={W}: spearman={rho:.4f}")

    # D. ¿qué predice el umbral óptimo? (ESN, 58 archivos, LOO)
    rows = []
    for rel, wins in labels.items():
        ts, sc = load_det(ESN, rel)
        wl = [(dp.parse(a), dp.parse(b)) for a, b in wins]
        alist = sw.calcSweepScore(list(ts), list(sc), list(wl), rel)
        best = max(sw.calcScoreByThreshold(alist), key=lambda x: x.score)
        raw = pd.read_csv(NAB / "data" / rel, usecols=["value"])["value"].to_numpy(float)
        alarms = (sc > 0.99).astype(int)
        runs = np.sum(np.diff(np.concatenate([[0], alarms, [0]])) == 1)
        rows.append({"thr": best.threshold, "largo": len(sc),
                     "curt": float(stats.kurtosis(raw)),
                     "empates": float(np.mean(sc == np.median(sc))),
                     "alarmas": float(alarms.sum() / max(len(sc), 1)),
                     "rachas": float(runs / max(alarms.sum(), 1))})
    df = pd.DataFrame(rows)
    for col in ("largo", "curt", "empates", "alarmas", "rachas"):
        rho, _ = stats.spearmanr(df["thr"], df[col])
        X = df[[col]].to_numpy(dtype=float)
        y = df["thr"].to_numpy(dtype=float)
        if np.std(X) < 1e-12:
            print(f"(D) thr~{col}: spearman={rho:.3f} R2_LOO=constante")
            continue
        try:
            pred = np.array([np.polyfit(np.delete(X, i, 0).ravel(),
                                        np.delete(y, i), 1)[0] * X[i, 0]
                             + np.polyfit(np.delete(X, i, 0).ravel(),
                                          np.delete(y, i), 1)[1]
                             for i in range(len(df))])
            ss = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
            print(f"(D) thr~{col}: spearman={rho:.3f} R2_LOO={ss:.3f}")
        except Exception as e:
            print(f"(D) thr~{col}: spearman={rho:.3f} R2_LOO=error-{type(e).__name__}")


if __name__ == "__main__":
    main()
