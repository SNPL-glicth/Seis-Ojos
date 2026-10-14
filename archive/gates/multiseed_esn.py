"""Multisemilla ESN (robustez por semilla del reservorio). Dos NAB completos."""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    out = Path(__file__).resolve().parents[2] / "results"
    runner = NABRunner(nab_root=nab)
    for seed in (7, 99):
        name = f"sixeyes_esn_seed{seed}_k24_tau2p0_w16000"

        def factory(s=seed):
            return StreamingPeakDecay(
                PercentileCalibrator(EchoStatePredictor(seed=s)),
                k_window=24, tau=2.0)

        scores = runner.evaluar(factory, name)
        (out / f"{name}.json").write_text(
            json.dumps({"seed": seed, "scores": scores}, indent=2, sort_keys=True))
        print(f"seed={seed}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)


if __name__ == "__main__":
    main()
