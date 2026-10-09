import math
import os
import numpy as np
import pandas as pd
import pytest

from src.detectors.base import Detector
from src.detectors.combinador import MaxOfDetectors, Combinado
from src.detectors.subespacio import SubspaceResidualDetector


class FakeDetector(Detector):
    def __init__(self, scores, raises=False):
        self.scores = list(scores)
        self.idx = 0
        self.call_count = 0
        self.raises = raises

    def update(self, value: float, timestamp: str) -> float:
        self.call_count += 1
        if self.raises:
            raise ValueError("Error simulado")
        if self.idx < len(self.scores):
            res = self.scores[self.idx]
            self.idx += 1
            return res
        return 0.0


def test_interfaz_y_scores_rango():
    combinado = Combinado()
    assert isinstance(combinado, Detector)

    np.random.seed(42)
    stream = np.random.randn(350)
    scores = []
    for i, val in enumerate(stream):
        s = combinado.update(float(val), str(i))
        scores.append(s)
        assert 0.0 <= s <= 1.0
        assert not math.isnan(s) and not math.isinf(s)


def test_detectores_falsos_maximo_y_sin_cortocircuito():
    # Detectores falsos con diferentes valores
    d1 = FakeDetector([0.2, 0.9, 0.1, float('nan'), float('inf')])
    d2 = FakeDetector([0.7, 0.4, 0.8, 0.3, 0.0])

    combinador = MaxOfDetectors([d1, d2])

    res0 = combinador.update(1.0, "0")
    assert res0 == 0.7
    assert d1.call_count == 1
    assert d2.call_count == 1

    res1 = combinador.update(2.0, "1")
    assert res1 == 0.9
    # d1 ya daba 0.9, pero d2 debe llamarse igual (sin cortocircuito)
    assert d1.call_count == 2
    assert d2.call_count == 2

    res2 = combinador.update(3.0, "2")
    assert res2 == 0.8
    assert d1.call_count == 3
    assert d2.call_count == 3

    # Prueba de NaN tratado como 0.0
    res3 = combinador.update(4.0, "3")
    assert res3 == 0.3
    assert d1.call_count == 4
    assert d2.call_count == 4

    # Prueba de Inf tratado como 0.0
    res4 = combinador.update(5.0, "4")
    assert res4 == 0.0
    assert d1.call_count == 5
    assert d2.call_count == 5


def test_anti_fuga_y_determinismo_combinado():
    # Calentamiento para PercentileCalibrator es 288
    k = 400
    n = 600

    np.random.seed(123)
    stream1 = np.random.randn(n)

    comb1 = Combinado()
    scores1 = [comb1.update(float(v), str(i)) for i, v in enumerate(stream1)]

    # Aseguramos que antes de k ya existan scores > 0
    scores_pre_k = np.array(scores1[:k])
    assert (scores_pre_k > 0).any(), "Debe haber scores > 0 antes de k (después del calentamiento)"

    # Alteramos todos los valores a partir de k
    stream2 = stream1.copy()
    stream2[k:] = stream2[k:] + 50.0

    comb2 = Combinado()
    scores2 = [comb2.update(float(v), str(i)) for i, v in enumerate(stream2)]

    # Anti-fuga: los scores hasta antes de k deben ser estrictamente idénticos
    assert np.allclose(scores1[:k], scores2[:k]), "Los scores antes del punto de alteración k difieren (fuga detectada)"

    # Determinismo: misma entrada produce exactamente los mismos scores
    comb3 = Combinado()
    scores3 = [comb3.update(float(v), str(i)) for i, v in enumerate(stream1)]
    assert np.allclose(scores1, scores3), "Combinado no es determinista"


def test_anti_fuga_datos_reales_smd():
    # Prueba anti-fuga con DATOS REALES: toma una serie SMD del set de ajuste,
    # altera todos los valores posteriores a la mitad y verifica que los scores
    # del SubspaceResidualDetector hasta la mitad son idénticos.
    smd_path = r"..\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U\182_SMD_id_5_Facility_tr_7174_1st_21230.csv"
    assert os.path.exists(smd_path), f"No se encontró el archivo de prueba SMD: {smd_path}"

    df = pd.read_csv(smd_path)
    valores = df.iloc[:, 0].values.astype(float)
    n = len(valores)
    mitad = n // 2

    # Ejecución 1: serie original
    det1 = SubspaceResidualDetector()
    scores1 = [det1.update(float(v), str(i)) for i, v in enumerate(valores)]

    # Ejecución 2: serie con valores posteriores a la mitad fuertemente alterados
    valores_alterados = valores.copy()
    valores_alterados[mitad:] = valores_alterados[mitad:] * 10.0 + 100.0

    det2 = SubspaceResidualDetector()
    scores2 = [det2.update(float(v), str(i)) for i, v in enumerate(valores_alterados)]

    # Verificación de identidad estricta hasta la mitad
    assert len(scores1) == n
    assert len(scores2) == n
    assert np.allclose(scores1[:mitad], scores2[:mitad]), "Fuga detectada en SubspaceResidualDetector sobre datos reales SMD"
    # Verificar que los scores posteriores a la alteración sí cambian
    assert not np.allclose(scores1[mitad:], scores2[mitad:]), "Los scores posteriores no reflejaron la alteración"
