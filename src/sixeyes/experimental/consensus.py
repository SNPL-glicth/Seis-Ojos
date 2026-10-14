"""Consenso por conteo k-of-n + confirmación de cresta (experimental).

Patrón portable de `earthgeckoSkyline` (`NAB/nab/detectors/earthgecko_skyline/`):
7 algoritmos, `CONSENSUS=5`, salida binaria. Lo que NO se copia: scans O(N²)
por paso, expiración fija de 12h tuneada in-sample ("best NAB score").

Diseño: 4 votantes binarios diversos por construcción (nivel, pendiente,
densidad, fase) → `score = votos/n`. La diversidad importa más que la
cantidad; el operador es CONTEO, jamás `max()` (autopsia del Combinado:
SMD 0.740→0.298 con `max`).

Confirmación de cresta: fracción <0.5 sin fracción ≥0.5 en [t−2, t−1] se
emite ×0.5 en vez de plena (atenúa ruido aislado sin suprimir ventanas NAB
de 100–300 puntos, que una supresión dura perdería). Retraso efectivo 0
(decisión en t con memoria de 2 pasos).

Todos los votantes se actualizan SIEMPRE cada paso (sin cortocircuito) para
no congelar estados internos.
"""

import math
from collections import deque
from typing import Callable

from sixeyes.detectors.base import Detector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.dual_scale import SeasonalDualScaleRobustZ
from sixeyes.detectors.ramp_gradient import CausalRampGradientTracker
from sixeyes.detectors.subespacio import SubspaceResidualDetector
from sixeyes.experimental.density_voter import HistogramDensityVoter

VOTE_THRESHOLD = 0.9
RAMP_THRESHOLD = 4.0
CONFIRM_WINDOW = 2
ATTENUATION = 0.5


def _level_voter() -> tuple[Detector, Callable[[float], float]]:
    det = PercentileCalibrator(SeasonalDualScaleRobustZ())
    return det, lambda s: 1.0 if s >= VOTE_THRESHOLD else 0.0


def _ramp_voter() -> tuple[Detector, Callable[[float], float]]:
    det = CausalRampGradientTracker()
    return det, lambda s: 1.0 if abs(s) >= RAMP_THRESHOLD else 0.0


def _density_voter() -> tuple[Detector, Callable[[float], float]]:
    det = HistogramDensityVoter()
    return det, lambda s: 1.0 if s >= 0.5 else 0.0


def _phase_voter() -> tuple[Detector, Callable[[float], float]]:
    det = PercentileCalibrator(SubspaceResidualDetector())
    return det, lambda s: 1.0 if s >= VOTE_THRESHOLD else 0.0


class CountConsensus(Detector):
    """Fracción de votos k/n con confirmación de cresta atenuada."""

    def __init__(
        self,
        confirm_window: int = CONFIRM_WINDOW,
        attenuation: float = ATTENUATION,
    ) -> None:
        makers = [_level_voter, _ramp_voter, _density_voter, _phase_voter]
        self._voters: list[tuple[Detector, Callable[[float], float]]] = [m() for m in makers]
        self._n = len(self._voters)
        self._past: deque[float] = deque(maxlen=confirm_window)
        self._attenuation = attenuation

    @property
    def n_voters(self) -> int:
        """Número de votantes del consenso."""
        return self._n

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        votes = 0.0
        for det, to_vote in self._voters:
            s = det.update(value, timestamp if timestamp else "2026-10-12T00:00:00")
            if math.isnan(s) or math.isinf(s):
                continue
            votes += to_vote(s)
        fraction = votes / self._n
        confirmed = fraction >= 0.5 or any(p >= 0.5 for p in self._past)
        out = fraction if confirmed else fraction * self._attenuation
        self._past.append(fraction)
        return min(max(out, 0.0), 1.0)

    def reset(self) -> None:
        """Reinicia votantes y memoria de confirmación."""
        for det, _ in self._voters:
            reset = getattr(det, "reset", None)
            if callable(reset):
                reset()
        self._past.clear()
