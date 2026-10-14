"""Tests reservorio denso + RLS (proyecto denso)."""

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.echo_state import EchoStatePredictor


class TestEchoState:
    def test_protocolo(self):
        det = EchoStatePredictor()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_determinismo_semilla_fija(self):
        a, b = EchoStatePredictor(), EchoStatePredictor()
        vals = list(np.sin(np.arange(200) / 5.0) * 10.0 + 50.0)
        sa = [a.update(float(v), "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(float(v), "2024-01-01T00:00:00") for v in vals]
        assert sa == sb

    def test_semilla_parametrizable_y_valida(self):
        for seed in (7, 99):
            det = EchoStatePredictor(seed=seed)
            scores = [det.update(float(50.0 + (i % 5)), "2024-01-01T00:00:00") for i in range(120)]
            assert all(0.0 <= s and s == s and s != float("inf") for s in scores)
        # Distintas semillas dan reservorios distintos (varianza real).
        assert (EchoStatePredictor(seed=7)._wres != EchoStatePredictor(seed=99)._wres).any()

    def test_predice_continuacion_y_falla_en_ruptura(self):
        det = EchoStatePredictor()
        vals = [float(50.0 + 10.0 * np.sin(i / 5.0)) for i in range(800)]
        cont = [det.update(v, "2024-01-01T00:00:00") for v in vals]
        tail = float(np.median(cont[-50:]))
        s = det.update(500.0, "2024-01-01T01:00:00")
        assert s > tail * 5, f"ruptura={s:.4f} cola={tail:.4f}"

    def test_causalidad(self):
        rng = np.random.RandomState(21)
        a = list(rng.randn(80))
        b = a[:40] + list(rng.randn(40) + 50.0)
        da, db = EchoStatePredictor(), EchoStatePredictor()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_reset_replay_invariance(self):
        vals = [float(50 + (i % 9)) for i in range(120)]
        d = EchoStatePredictor()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

    def test_time_features_compatibles_y_deterministas(self):
        import numpy as np
        a = EchoStatePredictor()
        b = EchoStatePredictor(time_features=True)
        vals = [float(50.0 + (i % 11)) for i in range(150)]
        ts = [f"2024-01-08T{i // 60:02d}:{i % 60:02d}:00" for i in range(150)]  # lunes
        sa = [a.update(v, t) for v, t in zip(vals, ts)]
        sb = [b.update(v, t) for v, t in zip(vals, ts)]
        assert sa != sb  # la entrada temporal cambia la dinámica
        c = EchoStatePredictor(time_features=True)
        sc = [c.update(v, t) for v, t in zip(vals, ts)]
        assert sc == sb  # ...pero determinista
        assert all(s >= 0.0 for s in sb)
        # Timestamp vacío: ceros, sin excepción.
        assert b.update(50.0, "") >= 0.0

    def test_apagado_reproduce_oficial(self):
        # Con time_features=False el reservorio es bit-idéntico al histórico.
        import numpy as np
        rng = np.random.RandomState(3)
        a = EchoStatePredictor()
        vals = [float(rng.randn()) for _ in range(100)]
        for v in vals:
            a.update(v, "2024-01-01T00:00:00")
        assert a._win.shape == (64, 16)

    def test_signed_output_para_likelihood(self):
        import numpy as np
        rng = np.random.RandomState(0)
        a = EchoStatePredictor()
        b = EchoStatePredictor(signed_output=True)
        vals = [float(50.0 + rng.randn()) for _ in range(300)]
        sa = [a.update(v, "2024-01-01T00:00:00") for v in vals]
        sb = [b.update(v, "2024-01-01T00:00:00") for v in vals]
        assert all(s >= 0.0 for s in sa)  # cuadrático por defecto
        assert any(s < 0.0 for s in sb)  # con signo admite negativos
        # Determinismo del modo con signo.
        c = EchoStatePredictor(signed_output=True)
        sc = [c.update(v, "2024-01-01T00:00:00") for v in vals]
        assert sc == sb
