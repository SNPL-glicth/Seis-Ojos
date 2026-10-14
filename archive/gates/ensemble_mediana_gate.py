"""Gate mediana-de-3: detección + in-sample + LOCO (compara contra media y single)."""
import sys
from pathlib import Path
import json

sys.path.insert(0, "src")
from sixeyes.experimental.rank_ensemble import RankEnsembleESN
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.evaluation.nab_runner import NABRunner


def factory():
    return StreamingPeakDecay(
        PercentileCalibrator(RankEnsembleESN(statistic="median")),
        k_window=24, tau=2.0)


def main():
    nab = (Path(__file__).resolve().parents[2].parent / "NAB").resolve()
    name = "sixeyes_ensemble_mediana_k24_w16000"
    runner = NABRunner(nab_root=nab)
    scores = runner.evaluar(factory, name)
    dest = Path(__file__).resolve().parents[2] / "results" / "sixeyes_ensemble_mediana.json"
    dest.write_text(json.dumps({"detector": "ensemble_mediana", "scores": scores},
                               indent=2, sort_keys=True))
    print(scores)


if __name__ == "__main__":
    main()
