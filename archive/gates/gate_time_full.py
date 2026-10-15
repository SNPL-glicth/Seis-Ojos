"""Gate: ESN(time_features=True) + Cal + PeakDecay + Spatial + Refractory.

B1 midio time_features en base ESN sola (47.43 vs 49.05, archivado). Este es un
experimento distinto: time_features en el FULL (bypass cubre picos, refractario
contiene FPs). Un solo intento, screen en seed 42 primero.
Criterio: s42 >= 60.81 para pasar a 3 semillas; media >= 60.68 para promover.
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

from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
DET = "sixeyes_time_full_s42"


def factory():
    esn = EchoStatePredictor(seed=42, time_features=True)
    cal = PercentileCalibrator(esn, max_history=16000)
    dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    spatial = SpatialSpikeBypass(dec, tolerance=0.05, warmup=100)
    return RefractorySuppressor(spatial, w_rest=80, trigger_threshold=0.990)


def main():
    scores = NABRunner(nab_root=NAB).evaluar(factory, DET)
    print(f"[scorer] {DET}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{DET}.json").write_text(
        json.dumps({"detector": DET, "seed": 42, "time_features": True,
                    "scores": scores}, indent=2, sort_keys=True), encoding="utf-8")
    import shutil
    src = NAB / "results" / DET / f"{DET}_standard_scores.csv"
    if src.is_file():
        shutil.copy(src, ROOT / "results" / f"{DET}_standard_scores.csv")
    print(f"[screen] s42={scores['standard']:.2f} vs full-s42=60.81 -> "
          f"{'PASA a 3 semillas' if scores['standard'] >= 60.81 else 'ARCHIVADO'}", flush=True)


if __name__ == "__main__":
    main()
