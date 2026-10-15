"""Analisis: que ventanas detectan los top-4 (ARTime, numenta, contextOSE, htmjava)
que nuestro full pipeline (s42) no detecta.

Para cada ventana: el score maximo dentro de la ventana para cada detector.
Reporta: ventanas que top-4 ven y nosotros no, con el score de cada uno.
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
DETS = {
    "ours_full_s42": "sixeyes_abl_full_s42",
    "ARTime": "ARTime",
    "numenta": "numenta",
    "contextOSE": "contextOSE",
    "htmjava": "htmjava",
}


def load(det):
    det_dir = NAB / "results" / det
    frames = {}
    for csv in det_dir.rglob(f"{det}_*.csv"):
        if csv.name.endswith("_standard_scores.csv") or "reward" in csv.name:
            continue
        rel = f"{csv.parent.name}/{csv.name[len(det) + 1:-4]}.csv"
        frames[rel] = pd.read_csv(csv, usecols=["timestamp", "anomaly_score"])
    thr = None
    agg = det_dir / f"{det}_standard_scores.csv"
    if agg.is_file():
        thr = float(pd.read_csv(agg)["Threshold"].iloc[0])
    return frames, thr


def main():
    wins = json.load(open(NAB / "labels" / "combined_windows.json", encoding="utf-8"))
    det = {k: load(v) for k, v in DETS.items()}
    print("umbrales:", {k: round(det[k][1], 5) if det[k][1] else None for k in DETS}, flush=True)

    top4_only = []
    for rel, wl in wins.items():
        scores = {}
        for name, (frames, thr) in det.items():
            key = next((k for k in frames if k == rel), None)
            if not key:
                scores[name] = None
                continue
            df = frames[key]
            tss = pd.to_datetime(df["timestamp"])
            sc = df["anomaly_score"].to_numpy(dtype=float)
            best = 0.0
            for a, b in wl:
                aa, bb = dp.parse(a), dp.parse(b)
                m = ((tss >= aa) & (tss <= bb)).to_numpy()
                if m.sum():
                    best = max(best, sc[m].max())
            scores[name] = best

        for i, (a, b) in enumerate(wl):
            ours = scores["ours_full_s42"] or 0.0
            top4_max = max(scores.get(d) or 0.0 for d in ("ARTime", "numenta", "contextOSE", "htmjava"))
            if top4_max > 0.9 and ours < 0.99:
                top4_only.append((rel, a, b, round(ours, 4), round(top4_max, 4)))

    print(f"\nventanas top-4 ven y nosotros no: {len(top4_only)}", flush=True)
    for rel, a, b, ours, top in top4_only:
        print(f"  {rel:55s} {a[:10]} ours={ours:.4f} top4_max={top:.4f}", flush=True)


if __name__ == "__main__":
    main()
