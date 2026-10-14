"""Reservorio denso + lectura RLS (base oficial del pipeline).

Ver diario (proyecto denso): NAB 48.62 vs 45.19 (+3.43), determinista.
`EchoStatePredictor` sustituye la rareza puntual por error de predicción
geométrico; el resto del pipeline (calibrador + PeakDecay) no cambia.

Responde al diagnóstico de las actas markovianas: la cuantización escalar
destruye la geometría antes de la memoria. Aquí no hay símbolos: el vector
de retardo (Takens, d=16) se proyecta a un reservorio recurrente fijo
(ESN, Nres=64, radio espectral 0.9) y una lectura lineal RLS predice el
siguiente punto. Anomalía = error cuadrático de predicción en espacio
latente, normalizado aguas abajo por el calibrador ECDF.

Diseño (constantes a priori):
- d=16, stride=1 (`DelayCoordinateEmbedding` reutilizado).
- Nres=64, semilla fija 42 (determinismo bit a bit), radio espectral 0.9
  (propiedad de eco), tanh, escala de entrada 1.0.
- Estandarización causal rodante (W=6400, sumas O(1)) antes del reservorio.
- RLS con olvido λ=0.999, P₀=I (delta=1.0). O(Nres²)=O(4096) flops/punto.

Causalidad: pronostica con el estado del pasado, puntúa el error y DESPUÉS
actualiza RLS y reservorio. Sin futuro en ningún punto.
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


class EchoStatePredictor(Detector):
    """Error cuadrático de predicción ESN-RLS (score crudo ≥0)."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        reservoir_size: int = RESERVOIR_SIZE,
        spectral_radius: float = SPECTRAL_RADIUS,
        seed: int = SEED,
    ) -> None:
        self._n = reservoir_size
        self._embed = DelayCoordinateEmbedding(dimension=embed_dim, stride=1)
        rng = np.random.RandomState(seed)
        self._win = rng.uniform(-1.0, 1.0, size=(reservoir_size, embed_dim)) * INPUT_SCALE
        w = rng.uniform(-1.0, 1.0, size=(reservoir_size, reservoir_size))
        radius = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self._wres = w * (spectral_radius / radius)
        self._h = np.zeros(reservoir_size)
        self._wout = np.zeros(reservoir_size)
        self._P = np.eye(reservoir_size) * RLS_DELTA
        self._lam = RLS_FORGET
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
        h_new = np.tanh(self._win @ vec + self._wres @ self._h)
        pred_z = float(self._wout @ h_new)
        err = (z - pred_z) ** 2
        # RLS DESPUÉS de puntuar (causalidad).
        h = h_new.reshape(-1, 1)
        Ph = self._P @ h
        denom = self._lam + float((h.T @ Ph)[0, 0])
        K = Ph / max(denom, 1e-12)
        self._wout = (self._wout.reshape(-1, 1) + K * (z - pred_z)).ravel()
        self._P = (self._P - (K @ h.T) @ self._P) / self._lam
        self._h = h_new
        self._ingest_norm(value)
        return max(float(err), 0.0)

    def reset(self) -> None:
        """Limpia reservorio, RLS, embedding y normalización."""
        self._embed.reset()
        self._h = np.zeros(self._n)
        self._wout = np.zeros(self._n)
        self._P = np.eye(self._n) * RLS_DELTA
        self._norm.clear()
        self._nsum = 0.0
        self._nsumsq = 0.0
