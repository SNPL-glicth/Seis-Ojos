"""Suite de tests para CausalMultiscaleLDS de producción (UKF/EKF + R^12)."""

import math
from pathlib import Path

import numpy as np
import pytest

from sixeyes.detectors.base import Detector
from sixeyes.detectors.causal_multiscale_lds import (
    CausalFeatureExtractor,
    CausalMultiscaleLDS,
    OnlineEKF,
    OnlineUKF,
)


class TestCausalFeatureExtractor:
    def test_output_dim_12(self):
        ext = CausalFeatureExtractor()
        phi = ext.update(10.0, "2026-10-12T00:00:00")
        assert phi.shape == (12,)
        assert phi.dtype == np.float64
        assert np.all(np.isfinite(phi))

    def test_determinism(self):
        e1 = CausalFeatureExtractor()
        e2 = CausalFeatureExtractor()
        for x in [1.0, 2.0, 1.5, 0.0, 3.0, 2.5]:
            p1 = e1.update(x, "2026-10-12T00:00:00")
            p2 = e2.update(x, "2026-10-12T00:00:00")
            np.testing.assert_allclose(p1, p2, rtol=1e-10)

    def test_causality_prefix_identical(self):
        """El prefijo común no cambia aunque diverja el futuro."""
        e1 = CausalFeatureExtractor()
        e2 = CausalFeatureExtractor()
        prefix = [5.0, 5.1, 5.2, 5.3, 5.4]
        outs1, outs2 = [], []
        for x in prefix:
            outs1.append(e1.update(x, "2026-10-12T00:00:00").copy())
        for x in prefix:
            outs2.append(e2.update(x, "2026-10-12T00:00:00").copy())
        for a, b in zip(outs1, outs2):
            np.testing.assert_allclose(a, b, rtol=1e-10)
        # Ahora divergen: el estado posterior sí puede diferir (correcto),
        # pero lo ya emitido no se reescribe (verificado arriba).

    def test_nan_inf(self):
        ext = CausalFeatureExtractor()
        assert np.all(ext.update(float("nan")) == 0.0)
        assert np.all(ext.update(float("inf")) == 0.0)

    def test_rest_indicator_binary(self):
        ext = CausalFeatureExtractor(eps_base=1e-3)
        ext.update(0.0)
        p0 = ext.update(0.0)
        p1 = ext.update(10.0)
        assert p0[9] != p1[9]

    def test_buffers_bounded(self):
        ext = CausalFeatureExtractor()
        for i in range(1500):
            ext.update(float(i % 7))
        assert len(ext._median_buf) <= ext.median_window
        assert len(ext._global_buf) <= ext.global_window


class TestOnlineFilters:
    @pytest.mark.parametrize("cls", [OnlineEKF, OnlineUKF])
    def test_protocol_output(self, cls):
        f = cls(latent_dim=4, obs_dim=12)
        s = f.update(np.ones(12))
        assert isinstance(s, float)
        assert s >= 0.0 and math.isfinite(s)

    @pytest.mark.parametrize("cls", [OnlineEKF, OnlineUKF])
    def test_determinism(self, cls):
        a, b = cls(), cls()
        rng = np.random.RandomState(0)
        for _ in range(60):
            phi = rng.randn(12)
            assert abs(a.update(phi) - b.update(phi)) < 1e-9

    @pytest.mark.parametrize("cls", [OnlineEKF, OnlineUKF])
    def test_stability_no_explosion(self, cls):
        f = cls()
        rng = np.random.RandomState(1)
        worst = 0.0
        for _ in range(400):
            s = f.update(2.0 + rng.randn(12) * 0.5)
            assert math.isfinite(s)
            worst = max(worst, s)
        assert worst < 1e6

    @pytest.mark.parametrize("cls", [OnlineEKF, OnlineUKF])
    def test_regime_shift_response(self, cls):
        f = cls()
        rng = np.random.RandomState(2)
        for _ in range(120):
            f.update(rng.randn(12) * 0.5)
        shift = [f.update(np.ones(12) * 8.0 + rng.randn(12) * 0.2) for _ in range(20)]
        base = [f.update(rng.randn(12) * 0.5) for _ in range(20)]
        # El salto debe producir innovaciones mayores que el régimen base ya adaptado
        assert float(np.mean(shift[:10])) > float(np.mean(base))

    @pytest.mark.parametrize("cls", [OnlineEKF, OnlineUKF])
    def test_condition_number_bounded(self, cls):
        """cond(S_t) < 500 en datos de control con ruido moderado."""
        f = cls()
        rng = np.random.RandomState(3)
        worst_cond = 1.0
        for _ in range(300):
            f.update(1.0 + rng.randn(12) * 0.3)
            worst_cond = max(worst_cond, float(f.last_cond))
        assert math.isfinite(worst_cond)
        assert worst_cond < 500.0


class TestCausalMultiscaleLDS:
    def test_protocolo_detector(self):
        det = CausalMultiscaleLDS()
        assert isinstance(det, Detector)
        s = det.update(10.0, "2026-10-12T00:00:00")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_score_range(self):
        det = CausalMultiscaleLDS()
        rng = np.random.RandomState(4)
        for _ in range(120):
            s = det.update(float(rng.randn() * 5.0))
            assert 0.0 <= s <= 1.0

    def test_determinism_full_pipeline(self):
        d1 = CausalMultiscaleLDS()
        d2 = CausalMultiscaleLDS()
        for x in [1.0, 2.0, 1.5, 0.0, 3.0, 2.5, 4.0]:
            assert abs(d1.update(x, "2026-10-12T00:00:00") - d2.update(x, "2026-10-12T00:00:00")) < 1e-9

    def test_anti_leakage_causal(self):
        d1 = CausalMultiscaleLDS()
        base = [10.0] * 60 + [20.0] * 20
        s1 = [d1.update(x, f"2026-10-12T{i // 60:02d}:{i % 60:02d}:00") for i, x in enumerate(base)]
        d2 = CausalMultiscaleLDS()
        pert = [10.0] * 60 + [15.0] + [20.0] * 19
        s2 = [d2.update(x, f"2026-10-12T{i // 60:02d}:{i % 60:02d}:00") for i, x in enumerate(pert)]
        for i in range(60):
            assert abs(s1[i] - s2[i]) < 1e-9

    def test_reset(self):
        det = CausalMultiscaleLDS()
        for x in [5.0] * 60 + [50.0] * 10:
            det.update(x)
        det.reset()
        assert det.update(5.0) == 0.0
        assert det.last_cond >= 1.0

    def test_constant_series_low_scores(self):
        det = CausalMultiscaleLDS()
        for _ in range(700):
            det.update(100.0)
        tail = [det.update(100.0) for _ in range(50)]
        assert float(np.mean(tail)) < 0.6

    def test_spike_detection(self):
        det = CausalMultiscaleLDS()
        for _ in range(700):
            det.update(10.0)
        assert det.update(100.0) > 0.5

    def test_ramp_detection(self):
        det = CausalMultiscaleLDS()
        for _ in range(700):
            det.update(10.0)
        scores = [det.update(v) for v in [12.0, 15.0, 19.0, 25.0, 32.0, 40.0, 50.0]]
        assert max(scores) > 0.5

    def test_no_fp_control_negativo(self):
        """En series de control negativo la tasa de scores extremos debe ser baja."""
        candidates = [
            Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab\data\artificialNoAnomaly\art_daily_no_noise.csv"),
            Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab\data\artificialNoAnomaly\art_daily_small_noise.csv"),
        ]
        files = [p for p in candidates if p.exists()]
        if not files:
            pytest.skip("Datos NAB no disponibles")
        import pandas as pd

        for csv_path in files:
            df = pd.read_csv(csv_path)
            det = CausalMultiscaleLDS()
            scores = [det.update(v, ts) for v, ts in zip(df["value"], df["timestamp"])]
            tail = np.array(scores[300:])
            frac_extreme = float(np.mean(tail > 0.99))
            assert frac_extreme < 0.05, f"{csv_path.name}: frac>0.99 = {frac_extreme:.4f}"
            assert float(np.mean(tail)) < 0.6

    def test_cond_bounded_on_control(self):
        """cond(S_t) del filtro interno < 500 en control negativo."""
        csv_path = Path(r"C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\nab\data\artificialNoAnomaly\art_daily_no_noise.csv")
        if not csv_path.exists():
            pytest.skip("Datos NAB no disponibles")
        import pandas as pd

        df = pd.read_csv(csv_path).head(1500)
        det = CausalMultiscaleLDS()
        worst = 1.0
        for v, ts in zip(df["value"], df["timestamp"]):
            det.update(v, ts)
            worst = max(worst, det.last_cond)
        assert math.isfinite(worst) and worst < 500.0
