import numpy as np
import math
from collections import deque
from sixeyes.detectors.base import Detector

# Canonical English names (production). Spanish aliases kept for compat.
WINDOW = 32
BUFFER_MAX = 1000
MIN_WINDOWS = 100
REFIT_EVERY = 50
TARGET_VARIANCE = 0.90
K_MAX = 8
# Aliases históricos (no usar en código nuevo).
VENTANA = WINDOW
MIN_VENTANAS = MIN_WINDOWS
REAJUSTE_CADA = REFIT_EVERY
VARIANZA_OBJETIVO = TARGET_VARIANCE

class SubspaceResidualDetector(Detector):
    """Detector de residuo de subespacio con compuerta causal de ingesta.

    NOTA DE INTEGRIDAD: el umbral histórico ``s < 0.70`` era un valor mágico
    ajustado in-sample sobre el Tuning-43 de TSB-AD sin validación externa.
    Por defecto el gating está DESACTIVADO (``gate_threshold=None`` = ingesta
    total). Pasar ``0.70`` explícito solo reproduce el experimento histórico.
    """

    def __init__(
                 self,
                 ventana=WINDOW,
                 buffer_max=BUFFER_MAX,
                 min_ventanas=MIN_WINDOWS,
                 reajuste_cada=REFIT_EVERY,
                 varianza_objetivo=TARGET_VARIANCE,
                 k_max=K_MAX,
                 gate_threshold: float | None = None):
        self.ventana = ventana
        self.buffer_max = buffer_max
        self.min_ventanas = min_ventanas
        self.reajuste_cada = reajuste_cada
        self.varianza_objetivo = varianza_objetivo
        self.k_max = k_max
        self.gate_threshold = gate_threshold

        self.history_raw = deque(maxlen=self.ventana)
        self.buffer = deque(maxlen=self.buffer_max)
        self.residuals = deque(maxlen=self.buffer_max)
        
        self.steps = 0
        self.model = {"mean": None, "components": None}

    def update(self, value: float, timestamp: str) -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0

        self.history_raw.append(value)
        if len(self.history_raw) < self.ventana:
            return 0.0

        x = np.array(self.history_raw, dtype=float)

        if self.model["components"] is not None:
            x_centered = x - self.model["mean"]
            if self.model["components"].shape[0] > 0:
                proj = x_centered @ self.model["components"].T
                x_reconstructed = proj @ self.model["components"]
            else:
                x_reconstructed = np.zeros_like(x_centered)
            
            r = np.linalg.norm(x_centered - x_reconstructed)
        else:
            r = 0.0

        # Calcula s
        pos_res = [res for res in self.residuals if res > 0]
        if not pos_res or r == 0:
            s = 0.0
        else:
            m = np.median(pos_res)
            s = float(r / (r + m))

        # Buffer Gating causal (opt-in): None = ingesta total, sin bloqueo.
        # 0.70 es heurística in-sample histórica, ver docstring.
        admit = True if self.gate_threshold is None else (s < self.gate_threshold)
        if admit:
            self.buffer.append(x)
            if r > 0:
                self.residuals.append(r)

        self.steps += 1

        if self.steps % self.reajuste_cada == 0 and len(self.buffer) >= self.min_ventanas:
            X = np.array(self.buffer)
            mean = X.mean(axis=0)
            X_c = X - mean
            
            U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            var_total = np.sum(S**2)
            
            if var_total > 0:
                var_exp = np.cumsum(S**2) / var_total
                k = np.searchsorted(var_exp, self.varianza_objetivo) + 1
                k = min(k, self.k_max)
                self.model["mean"] = mean
                self.model["components"] = Vt[:k]
            else:
                self.model["mean"] = mean
                self.model["components"] = np.zeros((0, X.shape[1]))

        return s

    def reset(self) -> None:
        """Reinicia buffers, residuos y modelo SVD."""
        self.history_raw.clear()
        self.buffer.clear()
        self.residuals.clear()
        self.steps = 0
        self.model = {"mean": None, "components": None}
