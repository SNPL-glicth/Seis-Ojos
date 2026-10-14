"""Detección + LOCO del ensemble por rangos (42/7/99) vs LOCO single-seed."""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.experimental.rank_ensemble import RankEnsembleESN
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner


def factory():
    return StreamingPeakDecay(PercentileCalibrator(RankEnsembleESN()), k_window=24, tau=2.0)


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    name = "sixeyes_ensemble_rangos_k24_w16000"
    runner = NABRunner(nab_root=nab)
    runner.generar_detecciones(factory, name)
    print(f"detecciones listas: {name}")


if __name__ == "__main__":
    main()
