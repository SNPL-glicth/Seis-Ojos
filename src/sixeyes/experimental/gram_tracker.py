"""
Detector de Anomalías en Streaming basado en Matriz de Gram Causal y Proyección de Subespacio (GramSubspaceTracker).

Combina dos canales ortogonales en el espacio de fases R^L:
- Canal A: Excursión Ortogonal del Atractor (residuo fuera del subespacio dominante U_k).
- Canal B: Colapso Log-Ratio de Grados de Libertad (pérdida de traza / singularidad).
"""

from collections import deque
import math
from typing import Optional
import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding


class GramSubspaceTracker(Detector):
    """Rastreador de subespacio y matriz de Gram causal en el espacio de fases R^L."""

    def __init__(
        self,
        dimension: int = 16,
        k_subspace: int = 2,
        stride: int = 1,
        alpha: float = 0.05,
        alpha_slow: float = 0.01,
        update_interval: int = 5,
        min_samples: int = 32,
        beta: float = 1.0,
        history_size: int = 200,
        epsilon: float = 1e-8,
    ) -> None:
        """Inicializa el detector GramSubspaceTracker.

        Args:
            dimension: Dimensión L del espacio de retardo.
            k_subspace: Número de vectores base dominantes U_k (k < L).
            stride: Paso de retardo temporal tau.
            alpha: Tasa de aprendizaje exponencial para mu_t y S_t.
            alpha_slow: Tasa lenta para la traza esperada de referencia.
            update_interval: Intervalo de pasos para actualización de base propia.
            min_samples: Período de calentamiento antes de emitir scores.
            beta: Ponderador relativo del Canal B (colapso de traza).
            history_size: Tamaño de buffer deslizante para calibración de MAD y mediana.
            epsilon: Regularizador numérico anti-cero.
        """
        if dimension <= k_subspace:
            raise ValueError(f"La dimensión L ({dimension}) debe ser estrictamente mayor que k ({k_subspace}).")
        if k_subspace < 1:
            raise ValueError("k_subspace debe ser al menos 1.")

        self.L = dimension
        self.k = k_subspace
        self.stride = stride
        self.alpha = alpha
        self.alpha_slow = alpha_slow
        self.update_interval = update_interval
        self.min_samples = min_samples
        self.beta = beta
        self.epsilon = epsilon

        self._embedder = DelayCoordinateEmbedding(dimension=dimension, stride=stride, normalize=False)
        self._mu: Optional[np.ndarray] = None
        self._S: Optional[np.ndarray] = None
        self._U_k: Optional[np.ndarray] = None
        self._mean_trace: float = 0.0

        self._residuals: deque[float] = deque(maxlen=history_size)
        self._discrepancies: deque[float] = deque(maxlen=history_size)
        self._steps = 0

    @property
    def is_ready(self) -> bool:
        """Indica si el modelo ha superado la fase de calentamiento."""
        return self._steps >= self.min_samples and self._U_k is not None

    def update(self, value: float, timestamp: str = "") -> float:
        """Procesa una nueva observación escalar y retorna un score de anomalía en [0, 1].

        Args:
            value: Observación escalar x_t.
            timestamp: Marca temporal asociada (opcional).

        Returns:
            float: Score de anomalía normalizado en [0.0, 1.0].
        """
        if math.isnan(value) or math.isinf(value):
            return 0.0

        vec, ready = self._embedder.update(value)
        if not ready or vec is None:
            return 0.0

        # Inicialización de media y covarianza si es el primer vector válido
        if self._mu is None:
            self._mu = vec.copy()
            self._S = np.zeros((self.L, self.L), dtype=np.float64)
            self._mean_trace = 0.0
            self._steps += 1
            return 0.0

        # Calentamiento inicial acumulativo
        if self._steps < self.min_samples:
            # Actualización recursiva de mu y S
            self._mu = (1.0 - self.alpha) * self._mu + self.alpha * vec
            vc = vec - self._mu
            self._S = (1.0 - self.alpha) * self._S + self.alpha * np.outer(vc, vc)
            current_trace = float(np.trace(self._S))
            if self._mean_trace <= self.epsilon:
                self._mean_trace = current_trace
            else:
                self._mean_trace = (1.0 - self.alpha_slow) * self._mean_trace + self.alpha_slow * current_trace

            self._steps += 1
            if self._steps == self.min_samples:
                self._update_subspace()
            return 0.0

        # 1. Proyección contra subespacio vigente (Canal A)
        vc = vec - self._mu
        if self._U_k is not None:
            coeff = vc @ self._U_k  # shape: (k,)
            proj = coeff @ self._U_k.T  # shape: (L,)
            r_t = float(np.linalg.norm(vc - proj))
        else:
            r_t = 0.0

        pos_r = [res for res in self._residuals if res > 1e-4]
        if r_t < 1e-4:
            s_A = 0.0
        elif not pos_r:
            s_A = 0.5
        else:
            m_r = float(np.median(pos_r))
            s_A = float(r_t / (r_t + m_r))

        # 2. Evaluación de Traza y Colapso (Canal B)
        current_trace = float(np.trace(self._S))
        if self._mean_trace > self.epsilon and current_trace >= 0.0:
            ratio = (self._mean_trace + self.epsilon) / (current_trace + self.epsilon)
            s_collapse = max(0.0, float(math.log(ratio)))
        else:
            s_collapse = 0.0

        # Normalizar Canal B a escala [0, 1] saturando a 3.0 (e^3 ≈ 20x colapso)
        s_B = min(1.0, s_collapse / 3.0)

        # 3. Fusión Conjunta Ortogonal
        denom_beta = 1.0 + self.beta
        score = float(math.sqrt((s_A**2 + self.beta * (s_B**2)) / denom_beta))
        score = max(0.0, min(1.0, float(score)))

        # 4. Buffer Gating con adaptación suave anti-congelamiento
        # Si score < 0.75: actualización completa
        # Si score >= 0.75: no contaminar residuos rápidos, pero adaptar lentamente (alpha * 0.1)
        if score < 0.75:
            self._mu = (1.0 - self.alpha) * self._mu + self.alpha * vec
            vc_new = vec - self._mu
            self._S = (1.0 - self.alpha) * self._S + self.alpha * np.outer(vc_new, vc_new)
            new_trace = float(np.trace(self._S))
            self._mean_trace = (1.0 - self.alpha_slow) * self._mean_trace + self.alpha_slow * new_trace

            if r_t > 1e-4:
                self._residuals.append(r_t)
            self._discrepancies.append(score)
        else:
            # Adaptación mínima para evitar bloqueo eterno ante cambio de régimen permanente
            alpha_slow_adapt = self.alpha * 0.05
            self._mu = (1.0 - alpha_slow_adapt) * self._mu + alpha_slow_adapt * vec
            vc_adapt = vec - self._mu
            self._S = (1.0 - alpha_slow_adapt) * self._S + alpha_slow_adapt * np.outer(vc_adapt, vc_adapt)

        # Recálculo periódico de la base U_k cada M pasos
        self._steps += 1
        if self._steps % self.update_interval == 0:
            self._update_subspace()

        return score

    def _update_subspace(self) -> None:
        """Actualiza la base propia U_k mediante descomposición de la matriz de dispersión S_t."""
        if self._S is None:
            return
        try:
            # eigh para matriz simétrica semidefinida positiva (L x L)
            w, V = np.linalg.eigh(self._S)
            # Extraer los k autovectores con mayores autovalores
            self._U_k = V[:, -self.k :]
        except np.linalg.LinAlgError:
            pass

    def reset(self) -> None:
        """Reinicia el detector a su estado inicial."""
        self._embedder.reset()
        self._mu = None
        self._S = None
        self._U_k = None
        self._mean_trace = 0.0
        self._residuals.clear()
        self._discrepancies.clear()
        self._steps = 0