"""Gate Paradigma III: Envolvente Causal Refractario (Dead-Time + Spatial Bypass) — NAB completo.

Pipeline:
RefractorySuppressor(w_rest=80, trigger=0.990)
  -> SpatialSpikeBypass(tolerance=0.05, warmup=100)
    -> StreamingPeakDecay(K=24, tau=2.0)
      -> PercentileCalibrator(W=16000)
        -> EchoStatePredictor(seed=42)
"""
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


def factory():
    esn = EchoStatePredictor(seed=42)
    cal = PercentileCalibrator(esn, max_history=16000)
    decay = StreamingPeakDecay(cal, k_window=24, tau=2.0)
    spatial = SpatialSpikeBypass(decay, tolerance=0.05, warmup=100)
    refractory = RefractorySuppressor(spatial, w_rest=80, trigger_threshold=0.990)
    return refractory


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    name = "sixeyes_refractory_w80_t990"
    print(f"Iniciando evaluación NAB completa para '{name}'...", flush=True)
    scores = NABRunner(nab_root=nab).evaluar(factory, name)
    out = Path(__file__).resolve().parents[2] / "results" / f"{name}.json"
    out.write_text(json.dumps({"scores": scores}, indent=2, sort_keys=True))
    print(f"Resultados oficiales guardados en {out}:", flush=True)
    print(scores, flush=True)


if __name__ == "__main__":
    main()
