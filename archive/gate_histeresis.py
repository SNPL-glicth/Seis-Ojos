"""Evaluación NAB histéresis: base + calibrador + histéresis (sin PeakDecay) vs 45.19."""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.detectors.dual_scale import SeasonalDualScaleRobustZ
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.experimental.hysteresis import HysteresisLatch
from sixeyes.evaluation.nab_runner import NABRunner


def factory():
    cal = PercentileCalibrator(SeasonalDualScaleRobustZ())
    latch = HysteresisLatch()

    class Pipe:
        def update(self, v, t=""):
            return latch.update(cal.update(v, t), t)

        def reset(self):
            cal.reset()
            latch.reset()

    return Pipe()


def main():
    r = NABRunner(nab_root=(Path(__file__).resolve().parents[1].parent / "NAB").resolve())
    scores = r.evaluar(factory, "sixeyes_histeresis_bA_k12_tau2p0_w16000")
    dest = Path(__file__).resolve().parents[1] / "results" / "sixeyes_histeresis.json"
    dest.write_text(json.dumps({"detector": "histeresis", "scores": scores}, indent=4, sort_keys=True))
    print(scores)


if __name__ == "__main__":
    main()
