"""Matriz base x capa, fila doble-escala: SeasonalDualScaleRobustZ + bypass.

Base determinista conocida 45.19 (sixeyes_dualscale_b15_k12_tau2p0_w16000).
Aqui: Dual -> Cal(W16000) -> PeakDecay(K=12, tau=2.0, mismos que su config) ->
SpatialSpikeBypass(tol=0.05, warmup=100). Sin refractory (matriz base x capa).
Una sola corrida NAB completa + scorer oficial.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.dual_scale import SeasonalDualScaleRobustZ
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
DET = "sixeyes_dual_bypass"


def factory():
    base = SeasonalDualScaleRobustZ(bucket_size_minutes=15)
    cal = PercentileCalibrator(base, max_history=16000)
    dec = StreamingPeakDecay(cal, k_window=12, tau=2.0)
    return SpatialSpikeBypass(dec, tolerance=0.05, warmup=100)


def main():
    runner = NABRunner(nab_root=NAB)
    scores = runner.evaluar(factory, DET)
    print(f"[scorer] {DET}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
    (ROOT / "results" / f"{DET}.json").write_text(
        json.dumps({"detector": DET, "scores": scores}, indent=2, sort_keys=True), encoding="utf-8")
    import shutil
    src = NAB / "results" / DET / f"{DET}_standard_scores.csv"
    if src.is_file():
        shutil.copy(src, ROOT / "results" / f"{DET}_standard_scores.csv")


if __name__ == "__main__":
    main()
