"""Evaluación NAB suma probabilística: OR-ruidoso + calibrador + PeakDecay vs 45.19."""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.experimental.noisy_or import NoisyOrFusion
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner


def factory():
    return StreamingPeakDecay(PercentileCalibrator(NoisyOrFusion()), k_window=12, tau=2.0)


def main():
    r = NABRunner(nab_root=(Path(__file__).resolve().parents[2].parent / "NAB").resolve())
    scores = r.evaluar(factory, "sixeyes_suma_probabilistica_bA_k12_tau2p0_w16000")
    dest = Path(__file__).resolve().parents[2] / "results" / "sixeyes_suma_probabilistica.json"
    dest.write_text(json.dumps({"detector": "suma_probabilistica", "scores": scores}, indent=4, sort_keys=True))
    print(scores)


if __name__ == "__main__":
    main()
