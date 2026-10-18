"""Residuo periódico causal multiescala (EXPERIMENTAL).

Ubicación: src/sixeyes/experimental/periodic_residual.py

Hipótesis: las 17 ventanas que solo ARTime detecta son subidas moderadas
dentro del rango histórico. Un perfil de fase por período puede separarlas
si la serie es periódica; si no lo es, la compuerta apaga el detector.

HIPERPARÁMETROS PRE-REGISTRADOS (fijados antes de correr NAB; no se tocan
mirando resultados):
    periods        = (48, 288, 336)
    alpha_min      = 0.10   (piso de la tasa de olvido del perfil por bin)
    clip           = 4.0    (winsorización de la actualización, en desv. medias abs.)
    min_visits     = 3      (visitas al bin antes de emitir score)
    gate_min       = 0.25   (R² mínimo del perfil periódico para activar)
    var_window     = 6400   (ventana EW de varianza global y MSE; = W_norm)
    dev_floor_frac = 0.05   (piso de la escala: 5% de la desv. estándar global)

Causalidad: el score de x_t usa solo el perfil construido con x_0..x_{t-1};
el perfil se actualiza DESPUÉS de emitir el score. Memoria O(sum(periods)),
acotada, no O(1). Costo por paso O(len(periods)).

Riesgo conocido: la fase es t mod P por índice de muestra (no por timestamp);
se desfasa si hay huecos o muestreo irregular.

Salida: score crudo NO acotado (r²), pensado para envolverse en su propio
PercentileCalibrator. Devuelve 0.0 cuando la compuerta está cerrada o no hay
suficientes visitas (PercentileCalibrator no ingiere raw == 0.0, protegiendo
el historial ECDF).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from sixeyes.detectors.base import Detector


class _PeriodState:
    __slots__ = ("period", "mu", "dev", "n", "mse", "n_mse")

    def __init__(self, period: int) -> None:
        self.period = period
        self.mu = [0.0] * period
        self.dev = [0.0] * period
        self.n = [0] * period
        self.mse = 0.0
        self.n_mse = 0


class PeriodicResidualDetector(Detector):
    """Score crudo = (|x_t - mu_fase| / escala_fase)² del mejor período, con compuerta."""

    def __init__(
        self,
        periods: Sequence[int] = (48, 288, 336),
        alpha_min: float = 0.10,
        clip: float = 4.0,
        min_visits: int = 3,
        gate_min: float = 0.25,
        var_window: int = 6400,
        dev_floor_frac: float = 0.05,
    ) -> None:
        if not periods or any(p < 2 for p in periods):
            raise ValueError("periods debe tener enteros >= 2")
        self._periods = tuple(int(p) for p in periods)
        self._alpha_min = alpha_min
        self._clip = clip
        self._min_visits = min_visits
        self._gate_min = gate_min
        self._var_window = var_window
        self._dev_floor_frac = dev_floor_frac
        self.reset()

    def reset(self) -> None:
        self._states = [_PeriodState(p) for p in self._periods]
        self._t = 0
        self._mean = 0.0
        self._var = 0.0
        self._n_glob = 0
        self._prev_x: float | None = None
        self._mse_naive = 0.0
        self._n_naive = 0

    def update(self, value: float, timestamp: str = "") -> float:
        x = float(value)
        if not math.isfinite(x):
            return 0.0

        t = self._t
        self._t += 1
        var_x = self._var
        std_x = math.sqrt(var_x) if var_x > 0.0 else 0.0

        # Error del baseline naive (lag 1, persistencia)
        if self._prev_x is not None:
            self._n_naive += 1
            a_naive = max(1.0 / self._n_naive, 1.0 / self._var_window)
            d_naive = x - self._prev_x
            self._mse_naive += a_naive * (d_naive * d_naive - self._mse_naive)
        
        # Base de comparación: mínimo entre varianza global y varianza de diferencias
        # Evita abrir en random walks donde MSE_fase < var(x) pero MSE_fase > MSE_naive
        ref_var = min(var_x, self._mse_naive) if self._mse_naive > 1e-12 and var_x > 1e-12 else var_x

        best_gate = -math.inf
        best_res = 0.0
        best_dev = 0.0

        for st in self._states:
            k = t % st.period
            n = st.n[k]

            if n >= 1:
                res = x - st.mu[k]
                # MSE del perfil (pre-actualización): mide cuánto explica la fase
                st.n_mse += 1
                a_m = max(1.0 / st.n_mse, 1.0 / self._var_window)
                st.mse += a_m * (res * res - st.mse)
                if n >= self._min_visits and ref_var > 1e-12:
                    # R² respecto a la referencia más exigente (varianza o persistencia lag-1)
                    gate = 1.0 - st.mse / ref_var
                    if gate > best_gate:
                        best_gate, best_res, best_dev = gate, res, st.dev[k]

            # Actualización del perfil (después de usarlo), winsorizada
            if n == 0:
                st.mu[k] = x
                st.dev[k] = 0.0
            else:
                d = x - st.mu[k]
                if n >= self._min_visits and st.dev[k] > 0.0:
                    lim = self._clip * st.dev[k]
                    d = max(-lim, min(lim, d))
                a = max(1.0 / (n + 1), self._alpha_min)
                st.mu[k] += a * d
                st.dev[k] = (1.0 - a) * st.dev[k] + a * abs(d)
            st.n[k] = n + 1

        # Estadísticos globales (después de emitir)
        self._n_glob += 1
        a_g = max(1.0 / self._n_glob, 1.0 / self._var_window)
        delta = x - self._mean
        self._mean += a_g * delta
        self._var = (1.0 - a_g) * (self._var + a_g * delta * delta)
        self._prev_x = x

        if best_gate < self._gate_min:
            return 0.0
        scale = max(best_dev, self._dev_floor_frac * std_x, 1e-9)
        score = (best_res / scale) ** 2
        return min(score, 1e12)


class MaxFusion(Detector):
    """Máximo de scores ya calibrados en [0, 1] de varios detectores.

    Cada rama debe venir envuelta en su propio PercentileCalibrator para que
    los scores sean comparables (sin gamma que sintonizar).
    """

    def __init__(self, detectors: Sequence[Detector]) -> None:
        if not detectors:
            raise ValueError("MaxFusion necesita al menos un detector")
        self._detectors = list(detectors)

    def reset(self) -> None:
        for d in self._detectors:
            if hasattr(d, "reset") and callable(d.reset):
                d.reset()

    def update(self, value: float, timestamp: str = "") -> float:
        # Se llama a todas las ramas siempre (cada una mantiene estado causal)
        scores = [d.update(value, timestamp) for d in self._detectors]
        return max(scores)
