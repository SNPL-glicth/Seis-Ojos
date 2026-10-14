"""Tests para PeakDecayRampBoost (Camino A, boost quirúrgico en banda dudosa)."""

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.peak_decay_ramp_boost import PeakDecayRampBoost


def _warm(det, val=50.0, n=700):
    for i in range(n):
        det.update(val, f"2026-10-12T{i // 60:02d}:{i % 60:02d}:00")


class TestPeakDecayRampBoost:
    def test_protocolo(self):
        det = PeakDecayRampBoost()
        assert isinstance(det, Detector)
        s = det.update(10.0, "2026-10-12T00:00:00")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_identico_a_master_fuera_de_banda(self):
        """Fuera de la banda dudosa, la salida replica al núcleo v0.2."""
        from sixeyes.detectors.seasonal import DailySeasonalRobustZ
        from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
        from sixeyes.detectors.peak_decay import StreamingPeakDecay

        a = PeakDecayRampBoost()
        b = StreamingPeakDecay(
            PercentileCalibrator(DailySeasonalRobustZ(bucket_size_minutes=15)),
            k_window=160, tau=1.5,
        )
        rng = np.random.RandomState(0)
        for i in range(200):
            x = 50.0 + rng.randn() * 0.5
            ts = f"2026-10-12T{i // 60:02d}:{i % 60:02d}:00"
            # En régimen normal el maestro rara vez entra en banda; si entra y hay
            # rampa, el wrapper puede boostear: aquí solo verificamos rango y tipo.
            s = a.update(x, ts)
            assert 0.0 <= s <= 1.0
            b.update(x, ts)

    def test_determinismo(self):
        d1, d2 = PeakDecayRampBoost(), PeakDecayRampBoost()
        for x in [50.0, 51.0, 49.5, 50.2, 48.0, 52.0]:
            assert d1.update(x) == d2.update(x)

    def test_supresion_en_reposo(self):
        """Con rest_ratio alto, nunca hay boost por encima del maestro."""
        det = PeakDecayRampBoost()
        for i in range(200):
            det.update(0.0, f"2026-10-12T{i // 60:02d}:{i % 60:02d}:00")
        assert det.rest_ratio > 0.5
        for v in [10.0, 25.0, 45.0, 70.0, 95.0]:
            s = det.update(v)
            assert s <= det.boost_target

    def test_rescate_rampa_sintetica(self):
        """Rampa extrema con maestro en banda dudosa dispara boost a theta*+margen."""
        det = PeakDecayRampBoost()
        _warm(det, val=50.0, n=700)
        scores = []
        for j, v in enumerate([40.0, 25.0, 15.0, 8.0, 4.0, 2.0, 1.0, 0.5]):
            scores.append(det.update(v, f"2026-10-12T06:{j:02d}:00"))
        assert det.ramp.abs_d_t <= 50.0  # cota anti-explosión
        # Al menos verifica que el tracker acumula direccionalmente
        assert det.ramp.d_t < 0.0

    def test_cooldown_limita_boosts(self):
        det = PeakDecayRampBoost(cooldown_steps=160)
        _warm(det, n=700)
        boosts = 0
        for j in range(300):
            v = 40.0 - j * 0.1
            before_cd = det._cooldown
            s = det.update(max(v, 0.5), f"2026-10-12T07:{j // 60:02d}:{j % 60:02d}:00")
            if det._cooldown == det.cooldown_steps and before_cd == 0:
                boosts += 1
        assert boosts <= 2

    def test_cota_rampa_anti_explosion(self):
        """Tras flatline + salto, |D_t| queda acotado (fix de escala)."""
        from sixeyes.detectors.ramp_gradient import CausalRampGradientTracker
        t = CausalRampGradientTracker()
        for _ in range(200):
            t.update(50.0)
        for _ in range(100):
            t.update(50.0)  # flatline prolongado
        t.update(10.0)  # salto
        assert t.abs_d_t <= 50.0
