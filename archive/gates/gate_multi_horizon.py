"""Gate Fase 16b: Multi-Horizon ESN — NAB completo.

Horizontes [1, 3, 6] fijos a priori, sin barrer.
Score = ‖[e₁, e₃, e₆]‖₂ → PercentileCalibrator(W=16000) → PeakDecay(K=24, τ=2.0).
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.experimental.multi_horizon_esn import MultiHorizonESN
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner


def factory():
    return StreamingPeakDecay(PercentileCalibrator(MultiHorizonESN()),
                              k_window=24, tau=2.0)


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    name = "sixeyes_multi_horizon_k24_w16000"
    scores = NABRunner(nab_root=nab).evaluar(factory, name)
    out = (Path(__file__).resolve().parents[2] / "results" / "sixeyes_multi_horizon.json")
    out.write_text(json.dumps({"scores": scores}, indent=2, sort_keys=True))
    print(scores, flush=True)


if __name__ == "__main__":
    main()
