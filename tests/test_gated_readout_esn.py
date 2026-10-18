"""Tests para GatedReadoutESN: equivalencia A/B, causalidad, telemetría e invariancias."""

import numpy as np
import pytest

from sixeyes.detectors.base import Detector
from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN
from sixeyes.experimental.gated_readout_esn import GatedReadoutESN


class TestGatedReadoutESN:
    def test_protocolo_detector(self):
        """GatedReadoutESN implementa la interfaz Detector."""
        det = GatedReadoutESN()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_equivalencia_ab_bit_a_bit_cuando_compuerta_desactivada(self):
        """Con enable_prefrontal_gate=False, la traza debe coincidir al 100% con PredictiveReadoutESN."""
        rng = np.random.RandomState(42)
        n = 500
        vals = list(np.sin(np.arange(n) / 10.0) * 15.0 + rng.randn(n) * 0.5 + 50.0)
        # Inyectar anomalía abrupta en t in [300, 350)
        for i in range(300, 350):
            vals[i] += 25.0

        base = PredictiveReadoutESN(seed=42)
        gated_off = GatedReadoutESN(enable_prefrontal_gate=False, seed=42)

        scores_base = [base.update(v, f"t_{i}") for i, v in enumerate(vals)]
        scores_off = [gated_off.update(v, f"t_{i}") for i, v in enumerate(vals)]

        assert len(scores_base) == len(scores_off)
        for i, (sb, so) in enumerate(zip(scores_base, scores_off)):
            assert sb == so, f"Discrepancia en paso {i}: base={sb} vs gated_off={so}"

        # Verificar estados de pesos finales
        assert np.allclose(base._rec.w, gated_off._rec.w)
        assert np.allclose(base._fc.w, gated_off._fc.w)
        assert np.allclose(base._h, gated_off._h)

    def test_compuerta_se_activa_en_anomalia_y_suprime_actualizaciones(self):
        """En meseta anómala, la compuerta debe activarse y suprimir actualizaciones RLS."""
        rng = np.random.RandomState(42)
        n = 400
        vals = list(np.sin(np.arange(n) / 8.0) * 10.0 + rng.randn(n) * 0.2 + 30.0)
        # Anomalía fuerte a partir del paso 200
        for i in range(200, 250):
            vals[i] += 40.0

        gated = GatedReadoutESN(enable_prefrontal_gate=True, seed=42)
        scores = [gated.update(v, f"t_{i}") for i, v in enumerate(vals)]

        telem = gated.get_telemetry()
        assert telem["enable_prefrontal_gate"] is True
        assert telem["total_steps"] == n
        assert telem["suppressed_updates"] > 0
        assert 0.0 < telem["suppressed_ratio"] < 1.0
        assert telem["gate_activations"] > 0

    def test_causalidad_estricta(self):
        """Los valores futuros no alteran los scores ni las decisiones de compuerta pasadas."""
        rng = np.random.RandomState(123)
        common = list(rng.randn(150) * 5.0 + 20.0)
        branch_a = common + list(rng.randn(100) * 5.0 + 20.0)
        branch_b = common + list(rng.randn(100) * 50.0 + 200.0)  # Divergencia masiva futura

        det_a = GatedReadoutESN(seed=99)
        det_b = GatedReadoutESN(seed=99)

        scores_a = [det_a.update(v, f"t_{i}") for i, v in enumerate(branch_a)]
        scores_b = [det_b.update(v, f"t_{i}") for i, v in enumerate(branch_b)]

        assert scores_a[:150] == scores_b[:150]

    def test_reset_limpia_estado_de_compuerta(self):
        """reset() restaura completamente contadores de telemetría y estado interno."""
        vals = [float(10.0 + i % 5) for i in range(120)]
        vals += [100.0] * 30  # disparar compuerta

        det = GatedReadoutESN(seed=42)
        for v in vals:
            det.update(v)

        assert det.suppressed_updates_count > 0
        det.reset()
        telem = det.get_telemetry()
        assert telem["total_steps"] == 0
        assert telem["suppressed_updates"] == 0
        assert telem["gate_activations"] == 0
        assert det.gate_active is False
