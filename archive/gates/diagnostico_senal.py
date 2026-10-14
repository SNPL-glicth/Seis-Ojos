"""Diagnósticos baratos con CSVs guardados (sin detectores nuevos).

(a) Cota superior por calibración: score NAB con umbral óptimo POR ARCHIVO.
(b) Métrica sin umbral: percentil del máximo en ventana vs distribución del archivo.
(c) Clasificación numérica de las 39 ventanas perdidas del ESN.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser as dp

sys.path.insert(0, "src")

NAB = Path(__file__).resolve().parents[2].parent / "NAB"
DET = "sixeyes_esn_k24_tau2p0_w16000"
sys.path.insert(0, str(NAB))
from nab.sweeper import Sweeper  # type: ignore

THR_GLOBAL = 0.99658055


def main():
    profiles = json.load(open(NAB / "config" / "profiles.json", encoding="utf-8"))
    labels = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    det_dir = NAB / "results" / DET

    # (a) umbral óptimo por archivo (perfil standard)
    cm = profiles["standard"]["CostMatrix"]
    sw = Sweeper(probationPercent=0.15, costMatrix=cm)
    total_best, total_glob = 0.0, 0.0
    for rel, wins in labels.items():
        rp = Path(rel)
        df = pd.read_csv(det_dir / rp.parent / f"{DET}_{rp.name}",
                         usecols=["timestamp", "anomaly_score"])
        ts = tuple(pd.to_datetime(df["timestamp"]).tolist())
        sc = tuple(float(x) for x in df["anomaly_score"].tolist())
        wl = tuple((dp.parse(a), dp.parse(b)) for a, b in wins)
        alist = sw.calcSweepScore(list(ts), list(sc), list(wl), rel)
        sweep = sw.calcScoreByThreshold(alist)
        total_best += max(t.score for t in sweep)
        _, mrow = sw.scoreDataSet(ts, sc, wl, rel, THR_GLOBAL)
        total_glob += mrow.score if mrow is not None else 0.0
    null_total = 0.0
    for rel, wins in labels.items():
        rp = Path(rel)
        ndf = pd.read_csv(NAB / "results" / "null" / rp.parent / f"null_{rp.name}",
                          usecols=["timestamp", "anomaly_score"])
        ts = tuple(pd.to_datetime(ndf["timestamp"]).tolist())
        sc = tuple(float(x) for x in ndf["anomaly_score"].tolist())
        wl = tuple((dp.parse(a), dp.parse(b)) for a, b in wins)
        _, mrow = sw.scoreDataSet(ts, sc, wl, rel, THR_GLOBAL)
        null_total += mrow.score if mrow is not None else 0.0
    tp_count = sum(len(v) for v in labels.values())
    perfect = tp_count * cm["tpWeight"]
    norm = lambda raw: 100.0 * (raw - null_total) / (perfect - null_total)
    print(f"(a) global={norm(total_glob):.2f}  por-archivo={norm(total_best):.2f}")

    # (b) percentil del máximo en ventana (39 perdidas del ESN)
    plost, pseen = [], []
    for rel, wins in labels.items():
        rp = Path(rel)
        df = pd.read_csv(det_dir / rp.parent / f"{DET}_{rp.name}",
                         usecols=["timestamp", "anomaly_score"])
        ts = pd.to_datetime(df["timestamp"])
        sc = df["anomaly_score"].to_numpy()
        for a, b in wins:
            aa, bb = dp.parse(a), dp.parse(b)
            idx = df.index[(ts >= aa) & (ts <= bb)].to_numpy()
            if len(idx) == 0:
                continue
            mx = sc[idx].max()
            pct = float(np.mean(sc <= mx) * 100.0)
            (pseen if mx >= THR_GLOBAL else plost).append(pct)
    plost = np.array(plost)
    print(f"(b) perdidas={len(plost)} vistas={len(pseen)}")
    print(f"    percentil del max en perdidas: med={np.median(plost):.2f} "
          f"p25={np.percentile(plost,25):.2f} "
          f">99.5: {np.mean(plost>99.5)*100:.0f}%  >99.9: {np.mean(plost>99.9)*100:.0f}%")

    # (c) clasificación de perdidas con datos crudos
    kinds: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for rel, wins in labels.items():
        rp = Path(rel)
        dfd = pd.read_csv(det_dir / rp.parent / f"{DET}_{rp.name}",
                          usecols=["timestamp", "anomaly_score"])
        ts = pd.to_datetime(dfd["timestamp"])
        sc = dfd["anomaly_score"].to_numpy()
        draw = pd.read_csv(NAB / "data" / rel, usecols=["value"])
        vals = draw["value"].to_numpy(dtype=float)
        for a, b in wins:
            aa, bb = dp.parse(a), dp.parse(b)
            idx = dfd.index[(ts >= aa) & (ts <= bb)].to_numpy()
            if len(idx) == 0 or sc[idx].max() >= THR_GLOBAL:
                continue
            win = vals[idx]
            ctx = np.concatenate([vals[max(0, idx[0]-500):idx[0]],
                                  vals[idx[-1]+1:idx[-1]+501]])
            if len(ctx) < 50:
                kind = "borde_serie"
            else:
                m_w, m_c = np.median(win), np.median(ctx)
                s_c = np.median(np.abs(ctx - m_c)) * 1.4826 + 1e-12
                zlev = abs(m_w - m_c) / s_c
                flat = np.mean(np.abs(np.diff(win)) <= 1e-9)
                v_w = np.std(win) + 1e-12
                v_c = np.std(ctx) + 1e-12
                if flat > 0.5:
                    kind = "flatline"
                elif zlev > 6:
                    kind = "cambio_nivel"
                elif v_w > 3 * v_c or v_c > 3 * v_w:
                    kind = "cambio_varianza"
                else:
                    kind = "sutil_contexto"
            kinds[kind] = kinds.get(kind, 0) + 1
            examples.setdefault(kind, []).append(f"{rel} [{a}..{b}]")
    print("(c) tipos en perdidas:", kinds)
    for k, v in examples.items():
        print(f"  {k}: {v[0]}" + (f" (+{len(v)-1})" if len(v) > 1 else ""))


if __name__ == "__main__":
    main()
