"""Predictor ESN multi-horizonte: error vectorial [t+1, t+3, t+6].

Extiende el ESN oficial con un readout RLS multi-output que predice
simultáneamente a tres horizontes temporales. El score es la norma
euclídea del vector de errores de predicción.

**Por qué esto es distinto y potencialmente mejor que el ESN base:**

1. Ataca la latencia (mediana 118 pts en NAB): si un cambio de tendencia
   ocurre en t+3, el error a horizonte 3 estalla en t (3 pasos antes).
   El horizonte 6 dispara AÚN antes. La norma combina señales tempranas.

2. Mantiene el readout aprendido: la lección de Δh (21.61) demostró que
   el mecanismo discriminativo del ESN es el readout RLS, no el estado
   crudo. Aquí EXPANDIMOS el readout a 3 cabezas lineales, cada una
   aprendida por RLS independiente.

3. Sensibilidad a rupturas suaves: un cambio de nivel lento da error
   pequeño a t+1 pero grande a t+6 (la predicción a largo plazo no
   puede seguir el drift). La norma vectorial lo amplifica.

Diseño (constantes a priori, sin barrer):
- Horizontes: [1, 3, 6] — fijos, cubren inmediato/corto/medio.
- Reservorio: idéntico al oficial (d=16, N=64, ρ=0.9, seed=42).
- RLS: λ=0.999 compartido, P₀=I por horizonte.
- Buffer circular de 6 valores z para targets retardados.
- Score: ‖[e₁, e₃, e₆]‖₂.

Causalidad: en cada paso t, los errores se computan SOLO contra targets
que ya llegaron (z_{t-h+1} predicho en t-h). El RLS se actualiza DESPUÉS.
O(3·Nres²) = O(12288) flops/punto.
"""

import math
from collections import deque

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.detectors.delay_embedding import DelayCoordinateEmbedding

EMBED_DIM = 16
RESERVOIR_SIZE = 64
SPECTRAL_RADIUS = 0.9
INPUT_SCALE = 1.0
RLS_DELTA = 1.0
RLS_FORGET = 0.999
NORM_WINDOW = 6400
SEED = 42
HORIZONS = (1, 3, 6)


class _HorizonHead:
    """Una cabeza de lectura RLS para un horizonte específico."""

    def __init__(self, n: int, lam: float = RLS_FORGET) -> None:
        self.wout = np.zeros(n)
        self.P = np.eye(n) * RLS_DELTA
        self.lam = lam

    def predict(self, h: np.ndarray) -> float:
        return float(self.wout @ h)

    def update_rls(self, h: np.ndarray, target: float) -> None:
        pred = float(self.wout @ h)
        err = target - pred
        h_col = h.reshape(-1, 1)
        Ph = self.P @ h_col
        denom = self.lam + float((h_col.T @ Ph)[0, 0])
        K = Ph / max(denom, 1e-12)
        self.wout = (self.wout.reshape(-1, 1) + K * err).ravel()
        self.P = (self.P - (K @ h_col.T) @ self.P) / self.lam

    def reset(self, n: int) -> None:
        self.wout = np.zeros(n)
        self.P = np.eye(n) * RLS_DELTA


class MultiHorizonESN(Detector):
    """Error de predicción multi-horizonte ESN-RLS (norma euclídea).

    Horizontes por defecto: [1, 3, 6]. El score es ‖[e₁, e₃, e₆]‖₂.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        reservoir_size: int = RESERVOIR_SIZE,
        spectral_radius: float = SPECTRAL_RADIUS,
        seed: int = SEED,
        horizons: tuple[int, ...] = HORIZONS,
    ) -> None:
        self._n = reservoir_size
        self._horizons = horizons
        self._max_h = max(horizons)
        self._embed = DelayCoordinateEmbedding(dimension=embed_dim, stride=1)
        in_dim = embed_dim
        rng = np.random.RandomState(seed)
        self._win = rng.uniform(-1.0, 1.0, size=(reservoir_size, in_dim)) * INPUT_SCALE
        w = rng.uniform(-1.0, 1.0, size=(reservoir_size, reservoir_size))
        radius = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self._wres = w * (spectral_radius / radius)
        self._h = np.zeros(reservoir_size)
        # Cabezas RLS por horizonte.
        self._heads = [_HorizonHead(reservoir_size) for _ in horizons]
        # Buffer circular: guarda (z_t, h_t) para actualizar targets retardados.
        # Necesitamos guardar los últimos max_h estados para poder actualizar
        # cada cabeza cuando su target llega.
        self._z_buffer: deque[float] = deque(maxlen=self._max_h + 1)
        self._h_buffer: deque[np.ndarray] = deque(maxlen=self._max_h + 1)
        # Estandarización causal.
        self._norm: deque[float] = deque(maxlen=NORM_WINDOW)
        self._nsum = 0.0
        self._nsumsq = 0.0
        self._step = 0

    def _standardize(self, value: float) -> float:
        n = len(self._norm)
        if n < 2:
            return 0.0
        mean = self._nsum / n
        var = max(self._nsumsq / n - mean * mean, 0.0)
        std = math.sqrt(var)
        if std <= 1e-9:
            return 0.0
        return (value - mean) / std

    def _ingest_norm(self, value: float) -> None:
        if len(self._norm) == self._norm.maxlen:
            old = self._norm[0]
            self._nsum -= old
            self._nsumsq -= old * old
        self._norm.append(value)
        self._nsum += value
        self._nsumsq += value * value

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        z = self._standardize(value)
        vec, ready = self._embed.update(z, timestamp)
        if not ready or vec is None:
            self._ingest_norm(value)
            self._z_buffer.append(z)
            return 0.0

        # Forward del reservorio.
        h_new = np.tanh(self._win @ vec + self._wres @ self._h)

        # --- SCORE: norma del error vectorial multi-horizonte ---
        # Cada cabeza predice z_{t+h} usando h_t.
        # Pero como estamos en t, solo podemos evaluar errores de predicciones
        # hechas EN EL PASADO cuyo target YA LLEGÓ.
        # Es decir: la predicción hecha en paso (t - h) para target z_t.
        errors = []
        for i, h_val in enumerate(self._horizons):
            if len(self._h_buffer) >= h_val:
                # Estado del paso t-h.
                past_h = self._h_buffer[-h_val]
                pred = self._heads[i].predict(past_h)
                errors.append(z - pred)

        if errors:
            score = float(np.linalg.norm(errors))
        else:
            score = 0.0

        # RLS update DESPUÉS de puntuar (causalidad).
        # Actualiza cada cabeza con su target correspondiente.
        for i, h_val in enumerate(self._horizons):
            if len(self._h_buffer) >= h_val:
                past_h = self._h_buffer[-h_val]
                self._heads[i].update_rls(past_h, z)

        # Guardar estado actual en buffers.
        self._h_buffer.append(h_new.copy())
        self._z_buffer.append(z)
        self._h = h_new
        self._ingest_norm(value)
        self._step += 1
        return max(score, 0.0)

    def reset(self) -> None:
        """Limpia reservorio, todas las cabezas RLS, embedding y buffers."""
        self._embed.reset()
        self._h = np.zeros(self._n)
        for head in self._heads:
            head.reset(self._n)
        self._z_buffer.clear()
        self._h_buffer.clear()
        self._norm.clear()
        self._nsum = 0.0
        self._nsumsq = 0.0
        self._step = 0
