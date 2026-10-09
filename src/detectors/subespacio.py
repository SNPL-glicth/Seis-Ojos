import numpy as np 
import math
from collections import deque
from src.detectors.base import Detector
#ajustamos los valores predeterminados segun el articulo 
VENTANA = 32
BUFFER_MAX = 1000
MIN_VENTANAS = 100
REAJUSTE_CADA = 50
VARIANZA_OBJETIVO = 0.90
K_MAX = 8

class SubspaceResidualDetector(Detector):
    def __init__(self, 
                 ventana=VENTANA, 
                 buffer_max=BUFFER_MAX, 
                 min_ventanas=MIN_VENTANAS, 
                 reajuste_cada=REAJUSTE_CADA, 
                 varianza_objetivo=VARIANZA_OBJETIVO, 
                 k_max=K_MAX):
        self.ventana = ventana
        self.buffer_max = buffer_max
        self.min_ventanas = min_ventanas
        self.reajuste_cada = reajuste_cada
        self.varianza_objetivo = varianza_objetivo
        self.k_max = k_max

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

        # Buffer Gating (s < 0.70): no contaminar buffer de SVD ni residuos con anomalias
        if s < 0.70:
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
