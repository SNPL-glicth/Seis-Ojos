"""Detector de anomalías basado en z-score robusto sobre ventana deslizante."""

from collections import deque
import math
import statistics

DEFAULT_WINDOW_SIZE: int = 50
DEFAULT_SCALE_K: float = 3.0
MIN_WARMUP_OBSERVATIONS: int = 5
NORMAL_SCALE_MAD: float = 1.4826


class RollingRobustZ:
    """Detector de anomalías en streaming usando mediana y MAD sobre ventana móvil.

    Calcula el z-score robusto: |valor - mediana| / (1.4826 * MAD) sobre las
    observaciones previas y lo acota en [0, 1] mediante la función 1 - exp(-z / k).

    Comportamiento en casos borde:
    - Calentamiento: Devuelve 0.0 mientras la ventana tenga menos de
      MIN_WARMUP_OBSERVATIONS observaciones.
    - MAD igual a cero: Si el valor actual coincide con la mediana de la serie
      constante, devuelve 0.0; si difiere, devuelve 1.0 (anomalía máxima) para
      evitar división por cero.
    - Valores NaN o infinitos: Devuelve 0.0 y no se añaden a la ventana histórica.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        k: float = DEFAULT_SCALE_K,
        min_observations: int = MIN_WARMUP_OBSERVATIONS,
    ) -> None:
        if window_size < 1:
            raise ValueError("El tamaño de ventana debe ser mayor o igual a 1.")
        if k <= 0:
            raise ValueError("El factor de escala k debe ser estrictamente positivo.")

        self._window_size = window_size
        self._k = k
        self._min_observations = min_observations
        self._window: deque[float] = deque(maxlen=window_size)

    def update(self, value: float, timestamp: str) -> float:
        """Devuelve un score en [0, 1], mayor valor significa más anómalo;
        el detector solo puede usar el pasado y el punto actual."""
        # Si el valor no es finito, se descarta para evitar corromper la distribución
        # estadística histórica y se retorna un score neutro de 0.0.
        if math.isnan(value) or math.isinf(value):
            return 0.0

        # Durante el calentamiento no hay suficientes datos para estimar la dispersión,
        # por lo que se retorna 0.0 mientras se llena la ventana mínima.
        if len(self._window) < self._min_observations:
            self._window.append(value)
            return 0.0

        mediana = statistics.median(self._window)
        desviaciones = [abs(x - mediana) for x in self._window]
        mad = statistics.median(desviaciones)

        # Si MAD es cero (ventana constante), se evita división por cero: coincidir con
        # la constante histórica no es anómalo; cualquier discrepancia es anomalía total.
        if mad == 0.0:
            score = 0.0 if value == mediana else 1.0
        else:
            escala = NORMAL_SCALE_MAD * mad
            z = abs(value - mediana) / escala
            score = 1.0 - math.exp(-z / self._k)

        # La observación actual entra a la ventana solo después de haber sido puntuada
        # garantizando causalidad estricta y ausencia de fuga de información.
        self._window.append(value)

        return min(max(score, 0.0), 1.0)
