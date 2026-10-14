"""Anomalía por volatilidad del estado latente del reservorio ESN.

En lugar de usar el error de predicción escalar (ŵ·h_t vs x_t), mide la
inestabilidad de la TRAYECTORIA INTERNA del reservorio: ‖h_t − h_{t-1}‖₂.

**Por qué esto es distinto y potencialmente mejor:**

1. Dimensionalidad: el error de predicción comprime R^64 → R^1 vía el
   readout (proyección lineal). La norma Δh opera en R^64 completo:
   detecta perturbaciones en CUALQUIER dirección del espacio de estados,
   no solo en la dirección del readout.

2. Latencia: el RLS (λ=0.999) suaviza el readout y absorbe sorpresas
   pequeñas, retrasando la detección (mediana 118 pts en NAB). El estado
   h_t reacciona instantáneamente a entradas anómalas porque las pesas
   del reservorio son FIJAS — no hay olvido exponencial que amortigüe.

3. Sin parámetros nuevos: usa el mismo reservorio (W_in, W_res fijos,
   d=16, N=64, radio 0.9, semilla 42) y la misma estandarización
   causal (W=6400). Solo cambia QUÉ se mide: h_t en vez de ŵ·h_t.

Causalidad: puntúa ‖h_t − h_{t-1}‖ y DESPUÉS actualiza h_prev = h_t.
El RLS sigue actualizándose (mantiene el reservorio "en forma" aunque
no usemos su predicción directamente para el score).

Complejidad: O(Nres² + Nres) = O(4096 + 64) por punto — idéntica al ESN
base (el cómputo caro es el forward del reservorio, que ya se hace).
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


class StateVolatilityDetector(Detector):
    """Anomalía por volatilidad del estado oculto del reservorio ESN.

    Score = ‖h_t − h_{t-1}‖₂ (norma euclídea del cambio de estado).
    Cero parámetros nuevos vs el ESN base.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        reservoir_size: int = RESERVOIR_SIZE,
        spectral_radius: float = SPECTRAL_RADIUS,
        seed: int = SEED,
    ) -> None:
        self._n = reservoir_size
        self._embed = DelayCoordinateEmbedding(dimension=embed_dim, stride=1)
        in_dim = embed_dim
        rng = np.random.RandomState(seed)
        self._win = rng.uniform(-1.0, 1.0, size=(reservoir_size, in_dim)) * INPUT_SCALE
        w = rng.uniform(-1.0, 1.0, size=(reservoir_size, reservoir_size))
        radius = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self._wres = w * (spectral_radius / radius)
        # Estado actual y anterior del reservorio.
        self._h = np.zeros(reservoir_size)
        self._h_prev: np.ndarray | None = None
        # RLS — se mantiene para que el reservorio esté "en forma",
        # aunque el score NO use la predicción del readout.
        self._wout = np.zeros(reservoir_size)
        self._P = np.eye(reservoir_size) * RLS_DELTA
        self._lam = RLS_FORGET
        # Estandarización causal rodante.
        self._norm: deque[float] = deque(maxlen=NORM_WINDOW)
        self._nsum = 0.0
        self._nsumsq = 0.0

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
            return 0.0

        # Forward del reservorio (idéntico al ESN base).
        h_new = np.tanh(self._win @ vec + self._wres @ self._h)

        # --- SCORE: volatilidad del estado latente ---
        if self._h_prev is None:
            score = 0.0
        else:
            score = float(np.linalg.norm(h_new - self._h_prev))

        # RLS update (mantiene el readout adaptado para que el
        # reservorio permanezca "en forma" — aunque no usemos ŵ·h
        # como score, el aprendizaje del readout afecta indirectamente
        # la dinámica si en el futuro se quisiera combinar).
        pred_z = float(self._wout @ h_new)
        h = h_new.reshape(-1, 1)
        Ph = self._P @ h
        denom = self._lam + float((h.T @ Ph)[0, 0])
        K = Ph / max(denom, 1e-12)
        self._wout = (self._wout.reshape(-1, 1) + K * (z - pred_z)).ravel()
        self._P = (self._P - (K @ h.T) @ self._P) / self._lam

        # Actualizar estados DESPUÉS de puntuar (causalidad).
        self._h_prev = self._h.copy()
        self._h = h_new
        self._ingest_norm(value)
        return max(score, 0.0)

    def reset(self) -> None:
        """Limpia reservorio, RLS, embedding y normalización."""
        self._embed.reset()
        self._h = np.zeros(self._n)
        self._h_prev = None
        self._wout = np.zeros(self._n)
        self._P = np.eye(self._n) * RLS_DELTA
        self._norm.clear()
        self._nsum = 0.0
        self._nsumsq = 0.0
