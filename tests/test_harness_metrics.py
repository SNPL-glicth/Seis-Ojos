import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sixeyes.evaluation.tsb_runner import _get_tsb_metrics


def test_tsb_metrics_harness_contracts():
    """Verifica que el arnés no invierta AUC-ROC ni AUC-PR y responda a contratos teóricos."""
    rng = np.random.default_rng(42)
    n = 2500
    labels = np.zeros(n, dtype=int)
    # Anomalía contigua de 50 puntos (prevalencia = 2%)
    labels[500:550] = 1

    # 1. Detector aleatorio uniforme [0, 1]
    scores_rnd = rng.uniform(0.0, 1.0, size=n)
    metrics_rnd = _get_tsb_metrics(scores_rnd, labels, sliding_window=20)

    # Contrato teórico:
    # AUC-ROC debe estar centrado en 0.50
    # AUC-PR debe estar centrado en la prevalencia (~0.02)
    assert 0.45 <= metrics_rnd["AUC-ROC"] <= 0.55, f"AUC-ROC aleatorio anómalo: {metrics_rnd['AUC-ROC']}"
    assert 0.005 <= metrics_rnd["AUC-PR"] <= 0.08, f"AUC-PR aleatorio no converge a prevalencia: {metrics_rnd['AUC-PR']}"

    # 2. Detector perfecto (score = label)
    scores_perf = labels.astype(float)
    metrics_perf = _get_tsb_metrics(scores_perf, labels, sliding_window=20)
    assert metrics_perf["AUC-ROC"] == 1.0, f"Detector perfecto debe dar AUC-ROC=1.0, dio {metrics_perf['AUC-ROC']}"
    assert metrics_perf["AUC-PR"] == 1.0, f"Detector perfecto debe dar AUC-PR=1.0, dio {metrics_perf['AUC-PR']}"

    # 3. Detector invertido (peor que el azar)
    scores_inv = 1.0 - scores_perf
    metrics_inv = _get_tsb_metrics(scores_inv, labels, sliding_window=20)
    assert metrics_inv["AUC-ROC"] == 0.0, f"Detector invertido debe dar AUC-ROC=0.0, dio {metrics_inv['AUC-ROC']}"
