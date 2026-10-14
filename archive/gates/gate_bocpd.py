"""Gate Paradigma II: BOCPD (sorpresa predictiva bayesiana) — NAB completo.

Score = −log P(x_t | x_{1:t-1}) bajo mixture NIGamma sobre run-lengths.
→ PercentileCalibrator(W=16000) → PeakDecay(K=24, τ=2.0).
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.experimental.bocpd import BOCPDDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner


def factory():
    return StreamingPeakDecay(PercentileCalibrator(BOCPDDetector()),
                              k_window=24, tau=2.0)


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    name = "sixeyes_bocpd_k24_w16000"
    scores = NABRunner(nab_root=nab).evaluar(factory, name)
    out = Path(__file__).resolve().parents[2] / "results" / "sixeyes_bocpd.json"
    out.write_text(json.dumps({"scores": scores}, indent=2, sort_keys=True))
    print(scores, flush=True)


if __name__ == "__main__":
    main()
