"""Envolvente causal de retención de crestas y atenuación exponencial de mesetas."""

from collections import deque
import math
from typing import Optional

from sixeyes.detectors.base import Detector


# Defaults conservadores (curva suave pre-explosión; ver diario).
# K=160, tau=1.5 es óptimo in-sample NAB-58; pasar explícito solo para
# reproducir el score 42.54, sin claim de generalización.
DEFAULT_K_WINDOW = 12
DEFAULT_TAU = 2.0

class StreamingPeakDecay(Detector):
    """Envolvente de Supresión de No-Máximos (NMS) causal para streaming.

    Permite emitir el score de anomalía a máxima intensidad en el frente de ataque
    del choque o cuando se establece una nueva cresta local en la ventana histórica
    [t - k_window, t - 1].
    
    Atenúa exponencialmente las mesetas, réplicas y fases de relajación posteriores
    mediante un factor exp(-(t - t_pico) / tau), eliminando ráfagas redundantes
    de falsas alarmas sin perjudicar la detección del impacto inicial.

    Propiedades matemáticas:
    - Causalidad estricta O(1): dependiente exclusivamente del historial pasado.
    - Cero fuga del futuro: invariante ante cualquier observación posterior a t.
    """

    def __init__(
        self,
        detector: Detector,
        k_window: int = DEFAULT_K_WINDOW,
        tau: float = DEFAULT_TAU,
    ) -> None:
        """Inicializa el envolvente de supresión de picos.

        Args:
            detector: Instancia del detector subyacente a calibrar/envolver.
            k_window: Tamaño de la ventana histórica de memoria para crestas locales.
            tau: Constante de tiempo de relajación exponencial.
        """
        if k_window < 1:
            raise ValueError("k_window debe ser mayor o igual a 1.")
        if tau <= 0.0:
            raise ValueError("tau debe ser estrictamente positivo.")

        self._detector = detector
        self.k_window = k_window
        self.tau = tau
        self._history: deque[float] = deque(maxlen=k_window)
        self._t: int = 0
        self._t_peak: int = 0

    def update(self, value: float, timestamp: str = "") -> float:
        """Procesa una observación entrante en tiempo real.

        Args:
            value: Valor numérico de la serie temporal.
            timestamp: Marca de tiempo asociada (formato ISO o cadena estándar).

        Returns:
            Score atenuado en [0.0, 1.0].
        """
        raw_score = self._detector.update(value, timestamp)

        if raw_score is None or math.isnan(raw_score) or math.isinf(raw_score):
            return 0.0

        raw_score = max(0.0, min(1.0, float(raw_score)))

        t = self._t
        self._t += 1

        # Si el historial previo está vacío o el score actual supera estrictamente el máximo histórico [t-K, t-1]
        if not self._history or raw_score > max(self._history):
            self._t_peak = t
            score_out = raw_score
        else:
            decay = math.exp(-(t - self._t_peak) / self.tau)
            score_out = raw_score * decay

        # Ingesta estrictamente causal: se almacena para las observaciones t+1 en adelante
        self._history.append(raw_score)

        return max(0.0, min(1.0, float(score_out)))

    def reset(self) -> None:
        """Reinicia el envolvente a su estado inicial.

        Limpia ``_history``, ``_t`` y ``_t_peak`` y delega en el detector
        interno si este expone ``reset()``. Garantiza la invarianza
        ``reset() + replay(stream) == replay(stream)``.
        """
        self._history.clear()
        self._t = 0
        self._t_peak = 0
        inner_reset = getattr(self._detector, "reset", None)
        if callable(inner_reset):
            inner_reset()
