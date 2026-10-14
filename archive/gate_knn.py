"""Evaluación NAB k-NN: k-NN + calibrador + PeakDecay vs 45.19."""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.experimental.knn_detector import StreamingKNNDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner


def factory():
    return StreamingPeakDecay(PercentileCalibrator(StreamingKNNDetector()), k_window=12, tau=2.0)


def main():
    r = NABRunner(nab_root=(Path(__file__).resolve().parents[1].parent / "NAB").resolve())
    scores = r.evaluar(factory, "sixeyes_knn_retardos_bA_k12_tau2p0_w16000")
    dest = Path(__file__).resolve().parents[1] / "results" / "sixeyes_knn_retardos.json"
    dest.write_text(json.dumps({"detector": "knn_retardos", "scores": scores}, indent=4, sort_keys=True))
    print(scores)


if __name__ == "__main__":
    main()
