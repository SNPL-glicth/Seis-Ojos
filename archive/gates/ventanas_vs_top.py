"""Comparador de ventanas: variante Spatial (s42) vs ARTime vs Full (s42).

Para cada ventana oficial de NAB, pregunta: el score maximo dentro de la
ventana cruza el umbral optimo del detector? Construye el Venn de deteccion
y lista la brecha real: ventanas que ARTime si ve y spatial no (subidas
moderadas dentro del rango, p. ej. nyc_taxi, grok, TravelTime).

Umbrales: el optimo in-sample de cada detector (standard_scores.csv).
Uso: python archive/gates/ventanas_vs_top.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from dateutil import parser as dp

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
DETS = ["sixeyes_abl_spatial_s42", "ARTime", "sixeyes_abl_full_s42"]


def load(det):
    """rel -> DataFrame por punto + threshold optimo standard."""
    det_dir = NAB / "results" / det
    thr = None
    agg = det_dir / f"{det}_standard_scores.csv"
    if agg.is_file():
        thr = float(pd.read_csv(agg)["Threshold"].iloc[0])
    frames = {}
    for csv in det_dir.rglob(f"{det}_*.csv"):
        if csv.name.endswith("_standard_scores.csv") or "reward" in csv.name:
            continue
        rel = f"{csv.parent.name}/{csv.name[len(det) + 1:-4]}.csv"
        frames[rel] = pd.read_csv(csv, usecols=["timestamp", "anomaly_score"])
    return frames, thr


def main():
    wins = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    det = {d: load(d) for d in DETS}
    print("umbrales:", {d: round(det[d][1], 5) for d in DETS}, flush=True)

    ven = {d: set() for d in DETS}
    gap_only_artime = []
    per_cat = {}
    for rel, wl in wins.items():
        cat = rel.split("/")[0]
        per_cat.setdefault(cat, [0, 0, 0, 0])  # both, onlyA, onlyS, neither
        rp = Path(rel)
        for d in DETS:
            frames, thr = det[d]
            key = None
            for k in frames:
                if k.replace("\\", "/") == rel or k == rel:
                    key = k
                    break
            if key is None:
                continue
            g = frames[key]
            ts = pd.to_datetime(g["timestamp"])
            sc = g["anomaly_score"].to_numpy(dtype=float)
            for a, b in wl:
                aa, bb = dp.parse(a), dp.parse(b)
                m = ((ts >= aa) & (ts <= bb)).to_numpy()
                if m.sum() == 0:
                    continue
                hit = sc[m].max() >= thr
                ven[d].add((rel, a))
                if hit and d == "ARTime":
                    pass
        # clasificar por ventana
        frames_s, thr_s = det["sixeyes_abl_spatial_s42"]
        frames_a, thr_a = det["ARTime"]
        frames_f, thr_f = det["sixeyes_abl_full_s42"]
        ks = next((k for k in frames_s if k.replace("\\", "/") == rel), None)
        ka = next((k for k in frames_a if k.replace("\\", "/") == rel), None)
        kf = next((k for k in frames_f if k.replace("\\", "/") == rel), None)
        if not (ks and ka and kf):
            continue
        ts_s = pd.to_datetime(frames_s[ks]["timestamp"])
        sc_s = frames_s[ks]["anomaly_score"].to_numpy(dtype=float)
        sc_a = frames_a[ka]["anomaly_score"].to_numpy(dtype=float)
        sc_f = frames_f[kf]["anomaly_score"].to_numpy(dtype=float)
        ts_a = pd.to_datetime(frames_a[ka]["timestamp"])
        ts_f = pd.to_datetime(frames_f[kf]["timestamp"])
        for a, b in wl:
            aa, bb = dp.parse(a), dp.parse(b)
            ms = ((ts_s >= aa) & (ts_s <= bb)).to_numpy()
            ma = ((ts_a >= aa) & (ts_a <= bb)).to_numpy()
            mf = ((ts_f >= aa) & (ts_f <= bb)).to_numpy()
            if not (ms.sum() and ma.sum() and mf.sum()):
                continue
            hs = sc_s[ms].max() >= thr_s
            ha = sc_a[ma].max() >= thr_a
            hf = sc_f[mf].max() >= thr_f
            if ha and hs:
                per_cat[cat][0] += 1
            elif ha and not hs:
                per_cat[cat][1] += 1
                gap_only_artime.append((rel, a, b, round(float(sc_s[ms].max()), 4)))
            elif hs and not ha:
                per_cat[cat][2] += 1
            else:
                per_cat[cat][3] += 1

    print("\n=== Venn de ventanas (umbral optimo in-sample de cada detector) ===", flush=True)
    print(f"{'categoria':26s} {'A&S':>4s} {'soloA':>6s} {'soloS':>6s} {'ninguna':>7s}")
    tot = [0, 0, 0, 0]
    for c, v in per_cat.items():
        print(f"{c:26s} {v[0]:4d} {v[1]:6d} {v[2]:6d} {v[3]:7d}")
        for i in range(4):
            tot[i] += v[i]
    print(f"{'TOTAL':26s} {tot[0]:4d} {tot[1]:6d} {tot[2]:6d} {tot[3]:7d}")
    print(f"\nbrecha real (ARTime ve, spatial NO): {tot[1]} ventanas", flush=True)
    for rel, a, b, mx in gap_only_artime:
        print(f"  {rel:55s} {a[:10]} max_score_spatial={mx}", flush=True)


if __name__ == "__main__":
    main()
