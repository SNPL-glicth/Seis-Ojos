"""Gate Fase 16: Latent State Volatility — NAB completo + LOCO."""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.experimental.state_volatility import StateVolatilityDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner


def factory():
    return StreamingPeakDecay(PercentileCalibrator(StateVolatilityDetector()),
                              k_window=24, tau=2.0)


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    name = "sixeyes_state_volatility_k24_w16000"
    scores = NABRunner(nab_root=nab).evaluar(factory, name)
    (Path(__file__).resolve().parents[2] / "results" / "sixeyes_state_volatility.json"
     ).write_text(json.dumps({"scores": scores}, indent=2, sort_keys=True))
    print(scores, flush=True)


if __name__ == "__main__":
    main()
