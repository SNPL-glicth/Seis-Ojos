"""Gate STE: TransitionSurpriseDetector + Cal + PeakDecay + Spatial + Refractory.

STE reemplaza al ESN-error en el slot base; downstream identico (W16000, K24,
tau2.0, tol0.05, warmup100, W80, t0.99). Parametros STE a priori: K=32, lr=0.02,
eps=1.0, primer orden. Sin barrido.
Criterio pre-registrado: media >= 60.68 en 3 semillas, ninguna peor que su
full actual, LOCO pareado no regresa.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.transition.transition_scorer import TransitionSurpriseDetector
from sixeyes.evaluation.nab_runner import NABRunner

NAB = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab")
FULL = {42: 60.81, 7: 54.66, 99: 57.58}


def main():
    runner = NABRunner(nab_root=NAB)
    means = {}
    for seed in (42, 7, 99):
        det = f"sixeyes_ste_s{seed}"

        def factory(s=seed):
            ste = TransitionSurpriseDetector(n_states=32, lr=0.02, epsilon=1.0, seed=s)
            cal = PercentileCalibrator(ste, max_history=16000)
            dec = StreamingPeakDecay(cal, k_window=24, tau=2.0)
            spatial = SpatialSpikeBypass(dec, tolerance=0.05, warmup=100)
            return RefractorySuppressor(spatial, w_rest=80, trigger_threshold=0.990)

        scores = runner.evaluar(factory, det)
        print(f"[scorer] {det}: " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), flush=True)
        (ROOT / "results" / f"{det}.json").write_text(
            json.dumps({"detector": det, "seed": seed, "scores": scores},
                       indent=2, sort_keys=True), encoding="utf-8")
        import shutil
        src = NAB / "results" / det / f"{det}_standard_scores.csv"
        if src.is_file():
            shutil.copy(src, ROOT / "results" / f"{det}_standard_scores.csv")
        means[seed] = scores["standard"]
    import statistics as st
    m = st.mean(means.values())
    print(f"\n[resultado] media={m:.2f} (42={means[42]:.2f} 7={means[7]:.2f} 99={means[99]:.2f})", flush=True)
    peores = [s for s in means if means[s] < FULL[s]]
    print(f"[criterio] media >= 60.68? {'SI' if m >= 60.68 else 'NO'}; "
          f"semillas peores que su full: {peores if peores else 'ninguna'}", flush=True)


if __name__ == "__main__":
    main()
