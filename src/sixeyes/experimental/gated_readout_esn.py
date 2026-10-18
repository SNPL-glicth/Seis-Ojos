"""GatedReadoutESN: Plasticidad regulada prefrontalmente sobre PredictiveReadoutESN.

Extiende directamente PredictiveReadoutESN (evitando duplicar el reservorio, el embedding
y las cabezas RLS) implementando los hooks de extensión causal:
- _should_update_weights()
- _should_ingest_norm()

Principios de Diseño:
1. Causalidad Estricta: La compuerta evalúa el error/score contemporáneo frente a un
   umbral causal estimado online (EWMA de media y dispersión), sin acceso a etiquetas ni futuro.
2. Equivalencia A/B Bit a Bit: Si enable_prefrontal_gate=False, el comportamiento coincide
   al 100% con la línea base original PredictiveReadoutESN.
3. Telemetría y Auditoría: Expone contadores de pasos, activaciones de compuerta y ratios
   de actualización para auditar si bloquea indebidamente derivas legítimas.
"""

from __future__ import annotations

import math
from typing import Dict, Any

from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN


class GatedReadoutESN(PredictiveReadoutESN):
    """PredictiveReadoutESN con compuerta de plasticidad y escala prefrontal causal."""

    def __init__(
        self,
        enable_prefrontal_gate: bool = True,
        threshold_sigma: float = 3.0,
        warmup_steps: int = 100,
        alpha_ewma: float = 0.05,
        gate_norm_ingest: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.enable_prefrontal_gate = bool(enable_prefrontal_gate)
        self.threshold_sigma = float(threshold_sigma)
        self.warmup_steps = int(warmup_steps)
        self.alpha_ewma = float(alpha_ewma)
        self.gate_norm_ingest = bool(gate_norm_ingest)

        # Estado causal de la compuerta prefrontal
        self._total_steps = 0
        self._active_steps = 0
        self._gate_active = False
        self._suppressed_updates_count = 0
        self._gate_activations_count = 0

        # Estimadores de régimen normal en streaming
        self._score_mean = 0.0
        self._score_var = 1.0

    @property
    def gate_active(self) -> bool:
        """Indica si la compuerta bloqueó la plasticidad en el paso contemporáneo."""
        return self._gate_active

    @property
    def suppressed_updates_count(self) -> int:
        """Número acumulado de pasos donde se suprimió la actualización RLS."""
        return self._suppressed_updates_count

    @property
    def total_steps(self) -> int:
        """Total de llamadas recibidas por update."""
        return self._total_steps

    @property
    def active_steps(self) -> int:
        """Pasos procesados con embedding listo y reservorio activo."""
        return self._active_steps

    @property
    def suppressed_ratio(self) -> float:
        """Fracción de pasos activos donde la plasticidad fue suprimida."""
        return self._suppressed_updates_count / max(self._active_steps, 1)

    def update(self, value: float, timestamp: str = "") -> float:
        self._total_steps += 1
        return super().update(value, timestamp)

    def _should_update_weights(
        self, score: float, z: float, pred_rec: float, pred_fc: float
    ) -> bool:
        self._active_steps += 1

        if not self.enable_prefrontal_gate:
            self._gate_active = False
            return True

        # Warmup inicial del reservorio: aprendizaje libre sin bloqueo
        if self._active_steps <= self.warmup_steps:
            if math.isfinite(score):
                self._score_mean = score if self._active_steps == 1 else (
                    (1.0 - self.alpha_ewma) * self._score_mean + self.alpha_ewma * score
                )
                diff = score - self._score_mean
                self._score_var = (
                    (1.0 - self.alpha_ewma) * self._score_var + self.alpha_ewma * (diff * diff)
                )
            self._gate_active = False
            return True

        # Umbral causal adaptativo: media + k * std
        std = math.sqrt(max(self._score_var, 1e-6))
        threshold = self._score_mean + self.threshold_sigma * std

        if score > threshold:
            # Sorpresa predictiva excesiva: se declara régimen anómalo / perturbación.
            # Bloquea la plasticidad para evitar que los pesos absorban la anomalía.
            self._gate_active = True
            self._suppressed_updates_count += 1
            self._gate_activations_count += 1
            return False

        # Régimen normal o deriva asimilable: permite actualización y actualiza estadísticas
        self._gate_active = False
        if math.isfinite(score):
            self._score_mean = (1.0 - self.alpha_ewma) * self._score_mean + self.alpha_ewma * score
            diff = score - self._score_mean
            self._score_var = (1.0 - self.alpha_ewma) * self._score_var + self.alpha_ewma * (diff * diff)
        return True

    def _should_ingest_norm(self, score: float, value: float) -> bool:
        if not self.enable_prefrontal_gate or not self.gate_norm_ingest:
            return True
        # Si la compuerta está activa, no contamina la ventana móvil con la anomalía
        return not self._gate_active

    def reset(self) -> None:
        super().reset()
        self._total_steps = 0
        self._active_steps = 0
        self._gate_active = False
        self._suppressed_updates_count = 0
        self._gate_activations_count = 0
        self._score_mean = 0.0
        self._score_var = 1.0

    def get_telemetry(self) -> Dict[str, Any]:
        """Devuelve el estado de auditoría de la compuerta."""
        return {
            "enable_prefrontal_gate": self.enable_prefrontal_gate,
            "total_steps": self._total_steps,
            "active_steps": self._active_steps,
            "suppressed_updates": self._suppressed_updates_count,
            "suppressed_ratio": self.suppressed_ratio,
            "gate_activations": self._gate_activations_count,
            "current_score_mean": self._score_mean,
            "current_score_std": math.sqrt(max(self._score_var, 1e-6)),
        }
