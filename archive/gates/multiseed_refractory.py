"""Multisemilla Refractory ESN (robustez por semilla del reservorio). Dos NAB completos."""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.experimental.refractory_suppressor import RefractorySuppressor
from sixeyes.experimental.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.evaluation.nab_runner import NABRunner


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    out = Path(__file__).resolve().parents[2] / "results"
    runner = NABRunner(nab_root=nab)
    for seed in (7, 99):
        name = f"sixeyes_refractory_seed{seed}_w80_t990"

        def factory(s=seed):
            esn = EchoStatePredictor(seed=s)
            cal = PercentileCalibrator(esn, max_history=16000)
            decay = StreamingPeakDecay(cal, k_window=24, tau=2.0)
            spatial = SpatialSpikeBypass(decay, tolerance=0.05, warmup=100)
            refractory = RefractorySuppressor(spatial, w_rest=80, trigger_threshold=0.990)
            return refractory

        scores = runner.evaluar(factory, name)
        (out / f"{name}.json").write_text(
            json.dumps({"seed": seed, "scores": scores}, indent=2, sort_keys=True))
        print(f"seed={seed}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)


if __name__ == "__main__":
    main()
