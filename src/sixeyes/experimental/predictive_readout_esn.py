"""PredictiveReadoutESN: readout predictivo puro + variante híbrida (Paso 1 del plan TSB).

Diagnóstico que ataca (README §1.1, hallazgo estructural): el ESN oficial evalúa
``pred_z = w @ h_t`` con ``h_t = tanh(Win @ v_t + Wres @ h_{t-1})``, donde ``v_t``
contiene a la observación contemporánea ``z_t``. El score mide distorsión de
escala contemporánea (reconstrucción), no sorpresa de secuencia.

Este módulo añade una segunda cabeza RLS que pronostica ``z_t`` con el estado
del paso anterior: ``pred_t = w_fc @ h_{t-1}``. El error de pronóstico dispara
ante ruptura del patrón de transición incluso con amplitud dentro del rango
histórico, y no estalla por sí solo en fronteras estacionales cuando se mezcla
con la cabeza de reconstrucción (variante híbrida).

Diseño (constantes a priori, sin barrer; reservorio idéntico al oficial):
- Cabeza de reconstrucción: ``w_rec`` entrenada por RLS con regresor ``h_t``
  y objetivo ``z_t`` (bit a bit igual que ``EchoStatePredictor``).
- Cabeza de pronóstico: ``w_fc`` entrenada por RLS con regresor ``h_{t-1}``
  y objetivo ``z_t``.
- Score híbrido: ``alpha * e_rec^2 + (1 - alpha) * e_fc^2`` con
  ``alpha = alpha_recon``. ``alpha=1.0`` ≡ ESN oficial; ``alpha=0.0`` =
  pronóstico puro; defecto 0.5 (híbrida, mitiga fronteras estacionales).
- Anti-windup por cabeza (re-anclaje w=0, P=I), mismo criterio del oficial.
- Techo numérico de residuos (1e12) idéntico al oficial.

Causalidad: ambos scores se computan ANTES de cualquier actualización RLS o
del reservorio; ``h_{t-1}`` proviene estrictamente del paso anterior. Coste
O(2·Nres²) = O(8192) flops/punto, O(1) en memoria.
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
ERR_CAP = 1e12


class _RLSHead:
    """Cabeza de lectura RLS con anti-windup (w=0, P=I si diverge)."""

    __slots__ = ("w", "P", "n", "lam")

    def __init__(self, n: int, lam: float = RLS_FORGET) -> None:
        self.n = n
        self.lam = lam
        self.reset()

    def predict(self, h: np.ndarray) -> float:
        return float(self.w @ h)

    def update(self, h: np.ndarray, target: float, pred: float) -> None:
        if not math.isfinite(pred) or not bool(np.all(np.isfinite(self.P))):
            self.reset()
            return
        h_col = h.reshape(-1, 1)
        Ph = self.P @ h_col
        denom = self.lam + float((h_col.T @ Ph)[0, 0])
        K = Ph / max(denom, 1e-12)
        self.w = (self.w.reshape(-1, 1) + K * (target - pred)).ravel()
        self.P = (self.P - (K @ h_col.T) @ self.P) / self.lam

    def reset(self) -> None:
        self.w = np.zeros(self.n)
        self.P = np.eye(self.n) * RLS_DELTA


class PredictiveReadoutESN(Detector):
    """Score híbrido reconstrucción + pronóstico puro sobre un ESN idéntico al oficial.

    Args:
        alpha_recon: peso de la cabeza de reconstrucción. 1.0 ≡ ESN oficial
            (bit a bit), 0.0 = pronóstico puro, defecto 0.5 (híbrida).
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        reservoir_size: int = RESERVOIR_SIZE,
        spectral_radius: float = SPECTRAL_RADIUS,
        seed: int = SEED,
        alpha_recon: float = 0.5,
    ) -> None:
        if not 0.0 <= alpha_recon <= 1.0:
            raise ValueError("alpha_recon debe estar en [0, 1]")
        self._n = reservoir_size
        self._alpha = float(alpha_recon)
        self._embed = DelayCoordinateEmbedding(dimension=embed_dim, stride=1)
        rng = np.random.RandomState(seed)
        self._win = rng.uniform(-1.0, 1.0, size=(reservoir_size, embed_dim)) * INPUT_SCALE
        w = rng.uniform(-1.0, 1.0, size=(reservoir_size, reservoir_size))
        radius = max(abs(np.linalg.eigvals(w)).max(), 1e-9)
        self._wres = w * (spectral_radius / radius)
        self._h = np.zeros(reservoir_size)
        self._rec = _RLSHead(reservoir_size)
        self._fc = _RLSHead(reservoir_size)
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

    @staticmethod
    def _sq(resid: float) -> float:
        if math.isnan(resid):
            return 0.0
        if not math.isfinite(resid):
            return ERR_CAP
        return min(resid * resid, ERR_CAP)

    def _should_update_weights(self, score: float, z: float, pred_rec: float, pred_fc: float) -> bool:
        """Hook de extensión: decide si actualizar los pesos RLS en este paso (defecto: siempre)."""
        return True

    def _should_ingest_norm(self, score: float, value: float) -> bool:
        """Hook de extensión: decide si incorporar el valor actual a la normalización (defecto: siempre)."""
        return True

    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        z = self._standardize(value)
        vec, ready = self._embed.update(z, timestamp)
        if not ready or vec is None:
            self._ingest_norm(value)
            return 0.0

        h_prev = self._h
        # Score ANTES de actualizar nada (causalidad estricta).
        h_new = np.tanh(self._win @ vec + self._wres @ h_prev)
        pred_rec = self._rec.predict(h_new)
        if not math.isfinite(pred_rec) or not bool(np.all(np.isfinite(self._rec.P))):
            # Anti-windup idéntico al oficial: re-ancla y puntúa contra pred=0.
            self._rec.reset()
            pred_rec = 0.0
        pred_fc = self._fc.predict(h_prev)
        if not math.isfinite(pred_fc) or not bool(np.all(np.isfinite(self._fc.P))):
            self._fc.reset()
            pred_fc = 0.0
        e_rec = z - pred_rec
        e_fc = z - pred_fc
        score = self._alpha * self._sq(e_rec) + (1.0 - self._alpha) * self._sq(e_fc)

        # RLS condicionado al hook de extensión
        if self._should_update_weights(score, z, pred_rec, pred_fc):
            self._rec.update(h_new, z, pred_rec)
            self._fc.update(h_prev, z, pred_fc)

        self._h = h_new

        # Ingesta de normalización condicionada al hook de extensión
        if self._should_ingest_norm(score, value):
            self._ingest_norm(value)

        return max(float(score), 0.0)

    def reset(self) -> None:
        """Limpia reservorio, ambas cabezas RLS, embedding y normalización."""
        self._embed.reset()
        self._h = np.zeros(self._n)
        self._rec.reset()
        self._fc.reset()
        self._norm.clear()
        self._nsum = 0.0
        self._nsumsq = 0.0