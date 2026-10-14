"""Detección B1 (ESN + tiempo) para métrica AUC. Sin scorer."""
import sys
from pathlib import Path

sys.path.insert(0, "src")
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner


def factory():
    return StreamingPeakDecay(
        PercentileCalibrator(EchoStatePredictor(time_features=True)),
        k_window=24, tau=2.0)


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    NABRunner(nab_root=nab).generar_detecciones(factory, "sixeyes_esn_tiempo_k24_w16000")
    print("detecciones B1 listas", flush=True)


if __name__ == "__main__":
    main()
