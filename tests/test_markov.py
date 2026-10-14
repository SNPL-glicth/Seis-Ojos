"""Tests sorpresa de transición markoviana."""

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.markov_transition import (
    MarkovTransitionSurprise,
    quantize,
    K_SYMBOLS,
)


class TestQuantize:
    def test_cotas_fijas(self):
        # 7 cortes → 8 símbolos: (-inf,-3),[-3,-2),[-2,-1),[-1,-.5),
        # [-.5,.5),[.5,1),[1,2),[2,inf). Fronteras semiabiertas [a,b).
        assert quantize(-10.0) == 0
        assert quantize(-3.0) == 1
        assert quantize(-2.0) == 2
        assert quantize(-0.75) == 3
        assert quantize(0.0) == 4
        assert quantize(0.49) == 4
        assert quantize(0.5) == 5
        assert quantize(1.5) == 6
        assert quantize(1.99) == 6
        assert quantize(2.0) == 7
        assert quantize(10.0) == 7

    def test_ocho_simbolos(self):
        assert K_SYMBOLS == 8
        assert len({quantize(x) for x in np.linspace(-5, 5, 200)}) == 8


class TestMarkov:
    def test_protocolo(self):
        det = MarkovTransitionSurprise()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0  # warmup

    def test_causalidad(self):
        rng = np.random.RandomState(21)
        a = list(rng.randn(80))
        b = a[:40] + list(rng.randn(40) + 50.0)
        da, db = MarkovTransitionSurprise(), MarkovTransitionSurprise()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_transicion_rara_sorprende_mucho(self):
        # Lo que el diseño sí garantiza: saltar a un símbolo extremo tras
        # régimen ruidoso da sorpresa alta (bits). El congelamiento SOSTENIDO
        # NO sorprende (self-loop central común en ruido): falsado y
        # documentado, no afirmado.
        rng = np.random.RandomState(9)
        det = MarkovTransitionSurprise()
        base = [det.update(float(50.0 + rng.randn()), "2024-01-01T00:00:00") for _ in range(600)]
        pre = float(np.median(base[-50:]))
        s = det.update(500.0, "2024-01-01T01:00:00")
        assert s > pre, f"salto={s:.2f} pre={pre:.2f}"

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 9) for i in range(120)]
        d = MarkovTransitionSurprise()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

    def test_sin_futuro_en_conteos(self):
        import numpy as np
        rng = np.random.RandomState(4)
        det = MarkovTransitionSurprise()
        for v in rng.randn(50):
            det.update(float(v), "2024-01-01T00:00:00")
        antes = [row[:] for row in det._counts]
        assert any(c > 0.0 for row in antes for c in row)  # hay conteos
        det.update(500.0, "2024-01-01T01:00:00")
        # El punto 500 puntuó contra conteos previos (no se auto-absorbió):
        assert antes != det._counts  # sí actualizó después
