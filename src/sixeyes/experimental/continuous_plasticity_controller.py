"""ContinuousPlasticityController (Fase III, experimental).

Subclase de PredictiveReadoutESN: reutiliza embedding, reservorio, normalización,
cabezas RLS y anti-windup. Solo sustituye las cabezas por WeightedRLSHead y fija
su ganancia eta_t en el hook ``_should_update_weights`` (no modifica la base).

Señales causales en el paso t (x = observación cruda, z = estandarizada):

    d_t = x_t - x_{t-1}
    s_t = max(1.4826 * MAD(d_{t-W..t-1}),  mean|d_{t-W..t-1}|,  1e-6 * max(1, |x_{t-1}|))
    v_t = |d_t| / s_t                                   (shock cinemático)

    e_t = z_t - w_fc^T h_{t-1}                         (error de pronóstico a 1 paso)
    p_t = (1 - k) p_{t-1} + k * clip(e_t / (r_{t-1} + eps), -1, 1)
    q_t = |p_t|                                         (persistencia direccional)

    g_t   = max(sigmoid(a (v_t - v0)), rho * g_{t-1})   (shock con histéresis)
    eta_t = 1 - g_t (1 - q_t)          =>  q_t <= eta_t <= 1

Orden temporal (causalidad):
  1. La base predice con el estado anterior y calcula el score.
  2. Hook: v_t usa la escala s_t de incrementos ANTERIORES; p_t usa la escala
     r_{t-1} de errores ANTERIORES. Se obtiene eta_t.
  3. La base actualiza las cabezas RLS con eta_t (RLS ponderado).
  4. Post-paso: se incorporan d_t al buffer de MAD y |e_t| a r_t.

Notas de diseño:
  - El suelo mean|d| evita que una rampa sin ruido (MAD = 0) dispare shocks.
  - El suelo relativo 1e-6 * max(1, |x|) evita divisiones por cero en señales
    constantes.
  - rho y k se fijan a priori por la ventana: semivida = W / 2 pasos.
  - Una deriva legítima abrupta y una anomalía abrupta son indistinguibles
    causalmente en su inicio; este controlador no pretende resolverlo, solo
    retrasar la plasticidad un tiempo acotado tras un shock.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List

import numpy as np

from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN
from sixeyes.experimental.weighted_rls import WeightedRLSHead

MAD_TO_SIGMA = 1.4826
_EPS = 1e-12


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class ContinuousPlasticityController(PredictiveReadoutESN):
    """PredictiveReadoutESN con plasticidad continua regulada por shock y persistencia."""

    def __init__(
        self,
        enable_controller: bool = True,
        window: int = 50,
        shock_center: float = 6.0,
        shock_slope: float = 1.5,
        warmup_steps: int = 100,
        record_trace: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if window < 4:
            raise ValueError("window debe ser >= 4")
        self.enable_controller = bool(enable_controller)
        self.window = int(window)
        self.shock_center = float(shock_center)
        self.shock_slope = float(shock_slope)
        self.warmup_steps = int(warmup_steps)
        self.record_trace = bool(record_trace)
        half_life = self.window / 2.0
        self.decay = 0.5 ** (1.0 / half_life)  # rho: memoria del shock
        self.rate = 1.0 - self.decay           # k: persistencia y escala del error
        self.min_history = self.window // 2
        # Cabezas ponderadas: estado inicial idéntico (w = 0, P = delta I).
        self._rec = WeightedRLSHead(self._n)
        self._fc = WeightedRLSHead(self._n)
        self._init_controller_state()

    # ------------------------------------------------------------------ estado
    def _init_controller_state(self) -> None:
        self._dbuf: deque[float] = deque(maxlen=self.window)
        self._prev_x: float | None = None
        self._x_t: float = 0.0
        self._e_t: float | None = None
        self._r: float | None = None
        self._p = 0.0
        self._g = 0.0
        self._active_steps = 0
        self._last = {"eta": 1.0, "shock": 0.0, "persistence": 0.0, "g": 0.0, "active": False}
        self._trace: Dict[str, List[float]] = {k: [] for k in self._last}
        self._set_eta(1.0)

    def _set_eta(self, eta: float) -> None:
        self._rec.eta = eta
        self._fc.eta = eta

    # ---------------------------------------------------------------- señales
    def _shock(self) -> float:
        """v_t con escala de incrementos estrictamente anteriores a t."""
        if self._prev_x is None or len(self._dbuf) < self.min_history:
            return 0.0
        d = self._x_t - self._prev_x
        arr = np.fromiter(self._dbuf, dtype=float, count=len(self._dbuf))
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        s = max(
            MAD_TO_SIGMA * mad,
            float(np.mean(np.abs(arr))),
            1e-6 * max(1.0, abs(self._prev_x)),
        )
        return abs(d) / s

    # ------------------------------------------------------------------ pasos
    def update(self, value: float, timestamp: str = "") -> float:
        if math.isnan(value) or math.isinf(value):
            score = super().update(value, timestamp)  # 0.0 sin tocar estado
            self._record(active=False, eta=1.0, shock=0.0, q=0.0, g=self._g)
            return score
        self._x_t = float(value)
        self._e_t = None
        self._last = {"eta": 1.0, "shock": 0.0, "persistence": 0.0, "g": self._g, "active": False}
        self._set_eta(1.0)
        score = super().update(value, timestamp)
        self._post_step(float(value))
        if self.record_trace:
            for k, v in self._last.items():
                self._trace[k].append(float(v))
        return score

    def _record(self, active: bool, eta: float, shock: float, q: float, g: float) -> None:
        self._last = {"eta": eta, "shock": shock, "persistence": q, "g": g, "active": active}
        if self.record_trace:
            for k, v in self._last.items():
                self._trace[k].append(float(v))

    def _should_update_weights(
        self, score: float, z: float, pred_rec: float, pred_fc: float
    ) -> bool:
        self._active_steps += 1
        e = z - pred_fc
        self._e_t = e if math.isfinite(e) else None

        if not self.enable_controller:
            self._last.update(eta=1.0, active=True)
            self._set_eta(1.0)
            return True

        v = self._shock()
        if self._active_steps <= self.warmup_steps or self._r is None or self._e_t is None:
            self._g = 0.0
            self._last.update(eta=1.0, shock=v, persistence=abs(self._p), g=0.0, active=True)
            self._set_eta(1.0)
            return True

        u = max(-1.0, min(1.0, e / (self._r + _EPS)))
        self._p = (1.0 - self.rate) * self._p + self.rate * u
        q = min(1.0, abs(self._p))
        g_inst = _sigmoid(self.shock_slope * (v - self.shock_center))
        self._g = max(g_inst, self.decay * self._g)
        eta = 1.0 - self._g * (1.0 - q)
        eta = min(1.0, max(0.0, eta))
        self._last.update(eta=eta, shock=v, persistence=q, g=self._g, active=True)
        self._set_eta(eta)
        return True

    def _post_step(self, value: float) -> None:
        """Incorpora la observación actual a las estadísticas históricas (paso 4)."""
        if self._prev_x is not None:
            self._dbuf.append(value - self._prev_x)
        self._prev_x = value
        if self._e_t is not None:
            a = abs(self._e_t)
            self._r = a if self._r is None else (1.0 - self.rate) * self._r + self.rate * a

    # -------------------------------------------------------------- auditoría
    def reset(self) -> None:
        super().reset()
        self._init_controller_state()

    def get_trace(self) -> Dict[str, np.ndarray]:
        """Traza por llamada a update (alineada con la entrada) si record_trace=True."""
        return {k: np.asarray(v, dtype=float) for k, v in self._trace.items()}

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "enable_controller": self.enable_controller,
            "active_steps": self._active_steps,
            "decay_rho": self.decay,
            "rate_k": self.rate,
            "last": dict(self._last),
        }
