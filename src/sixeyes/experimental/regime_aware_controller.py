"""RegimeAwarePlasticityController (Fase V, experimental).

Extensión de ContinuousPlasticityController dotada de una capa superior de
Memoria Episódica causal (EpisodicRegimeMemory).

Regla de Decisión Causal:
    eta_t = 1 - g_t (1 - q_t)   (ley de plasticidad continua de Fase III)
    
    Modulación por Evidencia de Régimen:
    1. Si status == "KNOWN" (distancia <= 1.25 a un régimen previo conocido):
       El sistema reconoce la dinámica. Si se trata del régimen base nominal,
       la adaptación es plena (eta_t se preserva o se acelera la recuperación).
    2. Si status == "UNCERTAIN_QUARANTINE" (la señal se aparta de todo régimen conocido):
       El sistema NO asume que la persistencia q_t garantice legitimidad.
       La plasticidad se mantiene acotada a un techo estricto (eta_quarantine_cap = 0.05),
       impidiendo que los pesos RLS absorban mesetas anómalas no verificadas.
    3. Si status == "NEW_PROMOTED" (la señal persistió coherentemente durante T_confirm pasos):
       El estado se reconoce como nueva normalidad permanente y se restablece la plasticidad.

Garantías:
    - Paridad A/B: enable_controller=False replica bit a bit PredictiveReadoutESN.
    - Causalidad Estricta: Todas las decisiones usan h_{t-1}, x_{t-1}, e_t causalmente.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sixeyes.experimental.continuous_plasticity_controller import (
    ContinuousPlasticityController,
)
from sixeyes.experimental.episodic_regime_memory import (
    EpisodicRegimeMemory,
    RegimeSignature,
)
from sixeyes.experimental.scale_estimators import (
    WindowedMADEstimator,
    StreamingHuberEstimator,
)


class RegimeAwarePlasticityController(ContinuousPlasticityController):
    """Controlador de plasticidad continua guiado por memoria episódica causal."""

    def __init__(
        self,
        enable_memory: bool = True,
        quarantine_cap: float = 0.05,
        quarantine_steps: int = 250,
        scale_estimator_type: str = "mad",  # "mad" | "huber"
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.enable_memory = bool(enable_memory)
        self.quarantine_cap = float(quarantine_cap)
        self.quarantine_steps = int(quarantine_steps)
        self.scale_estimator_type = str(scale_estimator_type).lower()
        
        # Estimador de escala configurable
        if self.scale_estimator_type == "huber":
            self._scale_est = StreamingHuberEstimator(half_life=self.window / 2.0, clip_sigmas=2.5)
        else:
            self._scale_est = WindowedMADEstimator(window=self.window, min_history=self.min_history)

        # Capa de memoria episódica
        self.memory = EpisodicRegimeMemory(
            reservoir_dim=self._n,
            quarantine_steps=self.quarantine_steps,
            smoothing_gamma=0.05,
        )
        self._warmup_latents: List[np.ndarray] = []
        self._warmup_x: List[float] = []
        self._baseline_registered = False
        self._regime_status = "INITIALIZING"
        self._regime_dist = float("inf")
        self._regime_id: Optional[int] = None
        self._last_memory_info: Dict[str, Any] = {
            "status": "INITIALIZING",
            "distance": float("inf"),
            "regime_id": None,
        }

    def _init_controller_state(self) -> None:
        super()._init_controller_state()
        if hasattr(self, "_scale_est"):
            self._scale_est.reset()
        if hasattr(self, "memory"):
            self.memory.reset()
        self._warmup_latents = []
        self._warmup_x = []
        self._baseline_registered = False
        self._regime_status = "INITIALIZING"
        self._regime_dist = float("inf")
        self._regime_id = None
        self._last_memory_info = {
            "status": "INITIALIZING",
            "distance": float("inf"),
            "regime_id": None,
        }

    def _shock(self) -> float:
        """v_t calculado con el estimador de escala seleccionado."""
        if self._prev_x is None:
            return 0.0
        d = self._x_t - self._prev_x
        s = self._scale_est.update(d, prev_x=self._prev_x)
        return abs(d) / max(s, 1e-12)

    def _should_update_weights(
        self, score: float, z: float, pred_rec: float, pred_fc: float
    ) -> bool:
        # Llamar a la lógica base de ContinuousPlasticityController
        super()._should_update_weights(score, z, pred_rec, pred_fc)

        if not self.enable_controller or not self.enable_memory:
            return True

        # Observar el estado recurrente causal h (anterior al nuevo paso recurrente)
        h_current = self._h.copy()

        # Fase de cristalización del régimen base nominal durante el calentamiento
        if self._active_steps <= self.warmup_steps:
            if self._active_steps > self.warmup_steps // 2:
                self._warmup_latents.append(h_current)
                self._warmup_x.append(float(self._x_t))
            if self._active_steps == self.warmup_steps and self._warmup_latents:
                self.memory.register_regime(
                    name="Nominal_Regime_0",
                    x_samples=self._warmup_x,
                    h_samples=self._warmup_latents,
                    regime_id=0,
                )
                self._baseline_registered = True
            self._regime_status = "CALIBRATING_BASE"
            self._last_memory_info = {
                "status": self._regime_status,
                "distance": 0.0,
                "regime_id": 0 if self._baseline_registered else None,
            }
            return True

        # Consulta a la memoria episódica
        best_id, dist, status = self.memory.update(float(self._x_t), h_current)
        self._regime_id = best_id
        self._regime_dist = dist
        self._regime_status = status
        self._last_memory_info = {
            "status": status,
            "distance": dist,
            "regime_id": best_id,
        }

        # Obtener eta actual calculado por el controlador continuo
        current_eta = self._last["eta"]

        # APLICACIÓN DE LA REGLA DE DECISIÓN DE RÉGIMEN:
        if status == "UNCERTAIN_QUARANTINE":
            # Estado no cartografiado: capar la plasticidad para no absorber la anomalía
            eta_modulated = min(current_eta, self.quarantine_cap)
            self._last["eta"] = eta_modulated
            self._set_eta(eta_modulated)
        elif status == "KNOWN" and best_id == 0:
            # Regreso al régimen base nominal conocido: acelerar recuperación
            if self._g > 0.5 and dist <= 1.0:
                # Si todavía hay shock residual de retorno pero el atractor es idéntico a Casa,
                # disolver shock residual
                self._g *= 0.5

        return True

    def get_memory_telemetry(self) -> Dict[str, Any]:
        return {
            "enable_memory": self.enable_memory,
            "scale_estimator": self.scale_estimator_type,
            "regime_status": self._regime_status,
            "regime_id": self._regime_id,
            "regime_distance": self._regime_dist,
            "known_regimes_count": len(self.memory.regimes),
            "quarantine_cap": self.quarantine_cap,
            "quarantine_steps": self.quarantine_steps,
        }
