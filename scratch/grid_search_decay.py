import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path("../benchmarks/nab").resolve()))

import numpy as np
from detectors.base import Detector
from detectors.seasonal import DailySeasonalRobustZ
from detectors.calibrador_percentil import PercentileCalibrator
from evaluation.nab_runner import NABRunner


class StreamingPeakDecay(Detector):
    def __init__(self, detector: Detector, K: int = 6, tau: float = 2.0):
        self._detector = detector
        self.K = K
        self.tau = tau
        self._history = []
        self._t = 0
        self._t_peak = 0

    def update(self, value: float, timestamp: str = "") -> float:
        raw_s = self._detector.update(value, timestamp)
        if raw_s is None or np.isnan(raw_s) or np.isinf(raw_s):
            return 0.0

        t = self._t
        self._t += 1

        if not self._history or raw_s >= max(self._history):
            self._t_peak = t
            s_out = raw_s
        else:
            decay = float(np.exp(-(t - self._t_peak) / self.tau))
            s_out = raw_s * decay

        self._history.append(raw_s)
        if len(self._history) > self.K:
            self._history.pop(0)

        return float(s_out)


def run_grid():
    configs = [
        (4, 1.5),
        (6, 1.5),
        (6, 2.0),
        (8, 2.0),
    ]

    runner = NABRunner()
    summary = {}

    for K, tau in configs:
        name = f"sixeyes_peakdecay_k{K}_t{int(tau*10)}"
        print(f"\n>>> Running {name} (K={K}, tau={tau}) <<<")
        runner.generar_detecciones(
            lambda K=K, tau=tau: StreamingPeakDecay(PercentileCalibrator(DailySeasonalRobustZ()), K=K, tau=tau),
            name
        )
        res = runner.ejecutar_scorer_oficial(name)
        summary[name] = res
        print(f"Result {name}: Standard={res.get('standard'):.2f}, Low_FP={res.get('reward_low_FP_rate'):.2f}, Low_FN={res.get('reward_low_FN_rate'):.2f}")

    print("\n================ FINAL SUMMARY ================")
    for k, v in summary.items():
        print(f"{k:<30}: Standard={v.get('standard'):.2f} | Low_FP={v.get('reward_low_FP_rate'):.2f} | Low_FN={v.get('reward_low_FN_rate'):.2f}")


if __name__ == "__main__":
    run_grid()
