"""Tests boost condicional por volatilidad."""

from sixeyes.experimental.volatility_boost import VolatilityBoost


class TestVolatilityBoost:
    def test_calma_no_toca(self):
        vb = VolatilityBoost()
        for _ in range(1000):
            vb.update_raw(50.0, 0.3)
        assert vb.update_raw(50.0, 0.3) == 0.3

    def test_arranque_volatil_amplifica(self):
        vb = VolatilityBoost()
        for _ in range(1000):
            vb.update_raw(50.0, 0.2)
        saw_boost = False
        for v in [50.0, 70.0, 30.0, 80.0, 20.0, 90.0, 10.0, 95.0]:
            if vb.update_raw(v, 0.2) > 0.2:
                saw_boost = True
                break
        assert saw_boost, "no hubo boost ante arranque volátil"

    def test_acotado(self):
        vb = VolatilityBoost()
        for _ in range(1000):
            vb.update_raw(50.0, 0.9)
        assert vb.update_raw(500.0, 0.9) <= 1.0

    def test_reset(self):
        vb = VolatilityBoost()
        for _ in range(1000):
            vb.update_raw(50.0, 0.3)
        vb.reset()
        assert vb.update_raw(50.0, 0.3) == 0.3
