"""Tests para PredictiveReadoutESN (Paso 1 del plan TSB: readout predictivo)."""

import numpy as np
import pytest

from sixeyes.detectors.base import Detector
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.experimental.predictive_readout_esn import PredictiveReadoutESN


class TestPredictiveReadoutESN:
    def test_protocolo_detector(self):
        """Implementa la interfaz Detector."""
        det = PredictiveReadoutESN()
        assert isinstance(det, Detector)
        # Primeros puntos: warmup embedding → 0.0
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_determinismo_semilla_fija(self):
        """Dos instancias misma semilla → scores idénticos."""
        a, b = PredictiveReadoutESN(), PredictiveReadoutESN()
        vals = list(np.sin(np.arange(200) / 5.0) * 10.0 + 50.0)
        sa = [a.update(float(v), "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert sa == sb

    def test_alpha_1_replica_esn_oficial_bit_a_bit(self):
        """alpha_recon=1.0 debe coincidir exactamente con EchoStatePredictor."""
        vals = [float(50.0 + 10.0 * np.sin(i / 5.0) + 0.1 * i) for i in range(400)]
        base = EchoStatePredictor()
        hyb = PredictiveReadoutESN(alpha_recon=1.0)
        sb = [base.update(float(v), "2024-01-01T00:00:00") for v in vals]
        sh = [hyb.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert sb == sh

    def test_pronostico_puro_dispara_en_ruptura_de_patron(self):
        """Cambio de ritmo (no de amplitud) dispara la cabeza de pronóstico."""
        # Serie senoidal; a mitad insertamos el mismo rango pero invertido de fase.
        vals = [float(50.0 + 10.0 * np.sin(i / 5.0)) for i in range(400)]
        vals += [float(50.0 + 10.0 * np.sin(i / 5.0 + np.pi)) for i in range(400, 500)]
        det = PredictiveReadoutESN(alpha_recon=0.0)
        scores = [det.update(v, "2024-01-01T00:00:00") for v in vals]
        antes = float(np.median(scores[390:400]))
        despues = float(np.max(scores[400:410]))
        assert despues > antes, f"sin reacción a inversión de fase: {despues} vs {antes}"

    def test_hibrido_puntua_ruptura_mas_alto_que_reconstruccion_sola(self):
        """En inversión de fase, la mezcla híbrida supera al score puro base."""
        vals = [float(50.0 + 10.0 * np.sin(i / 5.0)) for i in range(400)]
        vals += [float(50.0 + 10.0 * np.sin(i / 5.0 + np.pi)) for i in range(400, 430)]
        base = EchoStatePredictor()
        hyb = PredictiveReadoutESN(alpha_recon=0.5)
        sb = [base.update(v, "2024-01-01T00:00:00") for v in vals]
        sh = [hyb.update(v, "2024-01-01T00:00:00") for v in vals]
        # En los pasos posteriores a la inversión, el pronóstico debe aportar.
        assert max(sh[400:430]) > max(sb[400:430])

    def test_scores_no_negativos_y_finitos(self):
        """Scores ≥ 0 y finitos en secuencia larga y ruidosa."""
        det = PredictiveReadoutESN()
        rng = np.random.RandomState(0)
        vals = list(rng.randn(600) * 100 + 50)
        scores = [det.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert all(s >= 0.0 and s == s and s != float("inf") for s in scores)

    def test_causalidad_estricta(self):
        """Divergencia futura no afecta scores pasados."""
        rng = np.random.RandomState(21)
        a = list(rng.randn(80))
        b = a[:40] + list(rng.randn(40) + 50.0)
        da = PredictiveReadoutESN()
        db = PredictiveReadoutESN()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_reset_replay_invariance(self):
        """reset() + replay == replay desde cero (ambas cabezas)."""
        vals = [float(50 + (i % 9)) for i in range(150)]
        d = PredictiveReadoutESN()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

    def test_nan_inf_producen_cero(self):
        """NaN e Inf no propagan."""
        det = PredictiveReadoutESN()
        assert det.update(float("nan")) == 0.0
        assert det.update(float("inf")) == 0.0
        assert det.update(float("-inf")) == 0.0

    def test_alpha_fuera_de_rango_rechazado(self):
        """alpha_recon fuera de [0, 1] es error de construcción."""
        with pytest.raises(ValueError):
            PredictiveReadoutESN(alpha_recon=1.5)
        with pytest.raises(ValueError):
            PredictiveReadoutESN(alpha_recon=-0.1)

    def test_variantes_alpha_difieren(self):
        """Alphas distintos generan trayectorias de score distintas."""
        vals = [float(50 + (i % 7)) for i in range(200)]
        a = PredictiveReadoutESN(alpha_recon=0.0)
        b = PredictiveReadoutESN(alpha_recon=0.5)
        c = PredictiveReadoutESN(alpha_recon=1.0)
        sa = [a.update(v, "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(v, "2024-01-01T00:00:00") for v in vals]
        sc = [c.update(v, "2024-01-01T00:00:00") for v in vals]
        assert sa != sb and sb != sc

    def test_warmup_embedding_cero(self):
        """Los primeros embed_dim pasos son 0.0 (embedding no listo)."""
        det = PredictiveReadoutESN()
        for i in range(15):
            assert det.update(float(50 + i), "2024-01-01T00:00:00") == 0.0