"""Tests predictor de contexto variable."""

import numpy as np

from sixeyes.detectors.base import Detector
from sixeyes.experimental.context_predictor import VariableOrderPredictor, MAX_NODES


class TestContextPredictor:
    def test_protocolo(self):
        det = VariableOrderPredictor()
        assert isinstance(det, Detector)
        assert det.update(10.0, "2024-01-01T00:00:00") == 0.0

    def test_ruptura_de_patron_sorprende(self):
        # Patrón 50,51,50,51... aprendido; romperlo debe sorprender más que
        # la mediana del régimen.
        det = VariableOrderPredictor()
        base = [det.update(50.0 + (i % 2), f"2024-01-01T00:{i % 60:02d}:00") for i in range(400)]
        pre = float(np.median(base[-50:]))
        s = det.update(80.0, "2024-01-01T01:00:00")
        assert s > pre, f"ruptura={s:.2f} pre={pre:.2f}"

    def test_contexto_largo_supera_par(self):
        # Secuencia donde los PARES son vulgares pero el TRIPLE es inédito:
        # 4,4,5 repetido; luego 4,4,4. Markov-1 ve 4→4 común; el contexto
        # (4,4)→5 aprendido hace a (4,4)→4 sorprendente.
        det = VariableOrderPredictor()
        for _ in range(150):
            for v in (4.0, 4.0, 5.0):
                det.update(v, "2024-01-01T00:00:00")
        s_novel = det.update(4.0, "2024-01-01T01:00:00")  # cierra (4,4)→4
        assert s_novel > 3.0, f"triple inédito poco sorprendente: {s_novel:.2f}"

    def test_nodos_acotados(self):
        det = VariableOrderPredictor(max_nodes=200)
        rng = np.random.RandomState(2)
        for v in rng.randn(3000):
            det.update(float(v * 10), "2024-01-01T00:00:00")
        assert det._n_nodes <= 200, f"nodos={det._n_nodes}"

    def test_causalidad(self):
        rng = np.random.RandomState(21)
        a = list(rng.randn(80))
        b = a[:40] + list(rng.randn(40) + 50.0)
        da, db = VariableOrderPredictor(), VariableOrderPredictor()
        sa = [da.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(a)]
        sb = [db.update(v, f"2024-01-01T00:{i:02d}:00") for i, v in enumerate(b)]
        assert sa[:40] == sb[:40]

    def test_reset_replay_invariance(self):
        vals = [50.0 + (i % 9) for i in range(120)]
        d = VariableOrderPredictor()
        s1 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        d.reset()
        assert d._n_nodes == 1
        s2 = [d.update(v, "2024-01-01T00:00:00") for v in vals]
        assert s1 == s2

    def test_max_nodes_default(self):
        assert MAX_NODES == 4096
