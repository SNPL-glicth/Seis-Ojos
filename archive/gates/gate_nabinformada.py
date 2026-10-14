"""Gate NAB-informada: ESN con min/max global por archivo (NO causal).

Mide cuánta brecha es información (rango global, como ARTime/HTM) vs
arquitectura. Etiqueta obligatoria en todo reporte. Una sola corrida.
"""
import sys
from pathlib import Path
import csv
import json

import pandas as pd
from dateutil import parser as dp

sys.path.insert(0, "src")
from sixeyes.experimental.global_norm_esn import GlobalNormESN
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
NAB = (HERE.parents[2].parent / "NAB").resolve()
NAME = "sixeyes_nabinformada_minmax"


def main():
    nab = NAB
    labels = json.load(open(nab / "labels" / "combined_windows.json", encoding="utf-8"))
    wins = {k: [(dp.parse(a), dp.parse(b)) for a, b in v] for k, v in labels.items()}
    out = nab / "results" / NAME
    for rel in wins:
        df = pd.read_csv(nab / "data" / rel, usecols=["timestamp", "value"])
        vals = df["value"].tolist()
        ts = df["timestamp"].astype(str).tolist()
        det = StreamingPeakDecay(PercentileCalibrator(GlobalNormESN()), k_window=24, tau=2.0)
        det._detector._detector.set_range(float(min(vals)), float(max(vals)))
        rows = []
        for v, t in zip(vals, ts):
            s = det.update(float(v), t)
            lab = 1 if any(a <= dp.parse(t) <= b for a, b in wins[rel]) else 0
            rows.append((t, v, s, lab))
        dest = out / Path(rel).parent / f"{NAME}_{Path(rel).name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "value", "anomaly_score", "label"])
            w.writerows(rows)
    print("detecciones NAB-informadas listas", flush=True)
    scores = NABRunner(nab_root=nab).ejecutar_scorer_oficial(NAME)
    (ROOT / "results" / "sixeyes_nabinformada_minmax.json").write_text(
        json.dumps({"detector": NAME, "nab_informada": True, "scores": scores},
                   indent=2, sort_keys=True))
    print(scores, flush=True)


if __name__ == "__main__":
    main()
