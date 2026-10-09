import json
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
    """Causal Streaming Peak Retention & Decay Envelope.
    
    Permite emitir score pleno en crestas locales y frentes de choque crecientes,
    mientras atenúa exponencialmente la meseta y cola de relajación posterior
    para eliminar ráfagas redundantes de falsas alarmas fuera de ventana.
    
    S_tilde_t = S_t si S_t >= max(S_{t-K:t-1})
                S_t * exp(-(t - t_pico) / tau) si S_t < max(recientes)
    """
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


def test_params(K: int, tau: float):
    print(f"\n=======================================================")
    print(f"Evaluando StreamingPeakDecay con K={K}, tau={tau}")
    print(f"=======================================================")
    runner = NABRunner()
    detector_name = f"sixeyes_peakdecay_k{K}_tau{int(tau*10)}"
    
    # Generate CSVs
    runner.generar_detecciones(
        lambda: StreamingPeakDecay(PercentileCalibrator(DailySeasonalRobustZ()), K=K, tau=tau),
        detector_name
    )
    
    # Score
    results = runner.ejecutar_scorer_oficial(detector_name)
    print(f"\nRESULTADOS OFICIALES NAB ({detector_name}):")
    for profile, score in results.items():
        print(f"  {profile:<25}: {score:.4f}")
    return results

if __name__ == "__main__":
    test_params(K=6, tau=2.0)
