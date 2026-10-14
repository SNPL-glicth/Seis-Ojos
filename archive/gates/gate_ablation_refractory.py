"""Ablaciones causales de RefractorySuppressor y SpatialSpikeBypass en NAB completo.

Evalúa:
1. sixeyes_ablation_spatial_only: ESN + PeakDecay + SpatialSpikeBypass (sin Refractory)
2. sixeyes_ablation_refractory_only: ESN + PeakDecay + RefractorySuppressor (sin SpatialSpikeBypass)
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.evaluation.nab_runner import NABRunner


def factory_spatial_only():
    esn = EchoStatePredictor(seed=42)
    cal = PercentileCalibrator(esn, max_history=16000)
    decay = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    return SpatialSpikeBypass(decay, tolerance=0.05, warmup=100)


def factory_refractory_only():
    esn = EchoStatePredictor(seed=42)
    cal = PercentileCalibrator(esn, max_history=16000)
    decay = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    return RefractorySuppressor(decay, w_rest=80, trigger_threshold=0.990)


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    out_dir = Path(__file__).resolve().parents[2] / "results"
    runner = NABRunner(nab_root=nab)

    # 1. Spatial only
    name_sp = "sixeyes_ablation_spatial_only"
    print(f"=== Evaluando {name_sp} ===", flush=True)
    scores_sp = runner.evaluar(factory_spatial_only, name_sp)
    (out_dir / f"{name_sp}.json").write_text(
        json.dumps({"detector": name_sp, "scores": scores_sp}, indent=2, sort_keys=True)
    )
    print(f"Resultados {name_sp}: {scores_sp}", flush=True)

    # 2. Refractory only
    name_ref = "sixeyes_ablation_refractory_only"
    print(f"\n=== Evaluando {name_ref} ===", flush=True)
    scores_ref = runner.evaluar(factory_refractory_only, name_ref)
    (out_dir / f"{name_ref}.json").write_text(
        json.dumps({"detector": name_ref, "scores": scores_ref}, indent=2, sort_keys=True)
    )
    print(f"Resultados {name_ref}: {scores_ref}", flush=True)


if __name__ == "__main__":
    main()
