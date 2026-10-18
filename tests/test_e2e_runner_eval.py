import csv
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "experiments" / "tsb_frozen_p05"))
from frozen_runner import factory, FIELDNAMES, append_rows_dict
import evaluate_frozen_p05
from sixeyes.evaluation.tsb_runner import _get_tsb_metrics


def test_e2e_runner_dictwriter_and_evaluate_contract():
    """Test de extremo a extremo: runner escribe CSV con DictWriter y evaluate_frozen_p05 lo procesa."""
    rng = np.random.default_rng(123)
    n = 3000
    labels = np.zeros(n, dtype=int)
    # 60 puntos anómalos (~2% prevalencia)
    labels[1000:1060] = 1

    sliding_window = 20

    # Simular 3 series sintéticas con brazos: P10, P05, random, zscore
    # Probamos contratos:
    # 1. 'random' debe dar AUC-ROC ~ 0.50 y AUC-PR ~ prevalencia
    # 2. 'P05' con un score casi perfecto debe ganar a 'P10'
    rows = []
    for s_idx, fam in enumerate(["FAM_A", "FAM_A", "FAM_B"]):
        s_name = f"synth_{s_idx}_{fam}.csv"
        # Brazo random
        rnd_scores = np.clip(rng.uniform(0.0, 1.0, size=n), 0.0, 1.0)
        m_rnd = _get_tsb_metrics(rnd_scores, labels, sliding_window=sliding_window)

        # Brazo zscore
        zs_scores = np.clip(rng.uniform(0.1, 0.9, size=n), 0.0, 1.0)
        m_zs = _get_tsb_metrics(zs_scores, labels, sliding_window=sliding_window)

        # Brazo P10 (base con rendimiento medio)
        p10_scores = np.clip(labels * 0.4 + rng.uniform(0.0, 0.6, size=n), 0.0, 1.0)
        m_p10 = _get_tsb_metrics(p10_scores, labels, sliding_window=sliding_window)

        # Brazo P05 (candidato con rendimiento superior)
        p05_scores = np.clip(labels * 0.8 + rng.uniform(0.0, 0.2, size=n), 0.0, 1.0)
        m_p05 = _get_tsb_metrics(p05_scores, labels, sliding_window=sliding_window)

        for arm, m in [("random", m_rnd), ("zscore", m_zs), ("P10", m_p10), ("P05", m_p05)]:
            rows.append({
                "series": s_name,
                "family": fam,
                "n_points": n,
                "arm": arm,
                "auc_roc": float(m["AUC-ROC"]),
                "auc_pr": float(m["AUC-PR"]),
                "vus_pr": float(m["VUS-PR"]),
            })

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "synth_eval.csv"
        append_rows_dict(csv_file, rows)

        # Verificar contenido escrito por DictWriter
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == FIELDNAMES
            read_rows = list(reader)
            assert len(read_rows) == 12

        # Cargar a través de evaluate_frozen_p05
        data, meta = evaluate_frozen_p05.load(str(csv_file))
        assert len(data) == 3
        for s in data:
            assert 0.45 <= data[s]["random"]["auc_roc"] <= 0.55, f"Random AUC-ROC anómalo en {s}"
            assert data[s]["random"]["auc_pr"] <= 0.08, f"Random AUC-PR anómalo en {s}"
            for arm in ("P10", "P05", "random", "zscore"):
                assert arm in data[s], f"Brazo {arm} no encontrado en {s}"
                assert np.isfinite(data[s][arm]["auc_roc"]), f"AUC-ROC no finito en {arm}"
            assert data[s]["P05"]["auc_roc"] > data[s]["P10"]["auc_roc"]
