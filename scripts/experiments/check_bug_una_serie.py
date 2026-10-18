"""Chequeo de ~10 min para localizar el bug de AUC en la ruta por lotes.
Adapta SOLO las 3 funciones marcadas con ADAPTAR. No toca el detector ni el criterio.

Uso:  python check_bug_una_serie.py <nombre_serie_con_random_AUC<0.05> <valor_AUC_guardado_en_CSV>
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from sixeyes.evaluation.tsb_runner import _get_tsb_metrics, _resolve_tsb_path

_tsb_path = str(_resolve_tsb_path())
if _tsb_path not in sys.path:
    sys.path.append(_tsb_path)
from sixeyes.detectors.aleatorio import RandomDetector

TSB_DIR = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'


def _cargar_df(nombre):
    # Replica EXACTO de run_single_series: pd.read_csv(fp).dropna()
    df = pd.read_csv(os.path.join(TSB_DIR, nombre)).dropna()
    return df


def cargar_etiquetas(nombre):
    # Igual que tsb_runner: labels = df['Label'].astype(int).to_numpy()
    df = _cargar_df(nombre)
    return df['Label'].astype(int).to_numpy()


def scores_random_del_lote(nombre, n):
    # Igual que el brazo "random" del lote: RandomDetector(seed=42), channel="first",
    # val = float(data[i, 0]), update(val, str(i)).
    data = _cargar_df(nombre).iloc[:, 0:-1].values.astype(float)
    det = RandomDetector(seed=42)
    return [det.update(float(data[i, 0]), str(i)) for i in range(len(data))]


def auc_del_lote(y, s):
    # Misma funcion de AUC que usa el lote: _get_tsb_metrics (clip + TSB_AD get_metrics,
    # cuyo AUC-ROC es sklearn roc_auc_score sobre los scores recortados).
    data = None
    nombre = globals().get('_nombre_actual')
    if nombre:
        data = _cargar_df(nombre).iloc[:, 0:-1].values.astype(float)
    if data is not None and data.shape[1] >= 1:
        ref = data if data.shape[1] == 1 else data[:, 0].reshape(-1, 1)
        from TSB_AD.utils.slidingWindows import find_length_rank
        sliding_window = find_length_rank(ref, rank=1)
    else:
        sliding_window = 100
    return _get_tsb_metrics(np.asarray(s, dtype=float), np.asarray(y, dtype=int), sliding_window)['AUC-ROC']


def main(nombre, auc_csv):
    globals()['_nombre_actual'] = nombre
    y = np.asarray(cargar_etiquetas(nombre)).astype(int)
    rng = np.random.default_rng(0)
    print("serie:", nombre, "| n =", len(y), "| frac etiquetas 1 =", round(y.mean(), 4))
    print("standalone (sklearn, scores iid):",
          [round(roc_auc_score(y, rng.random(len(y))), 3) for _ in range(5)])

    s = np.asarray(scores_random_del_lote(nombre, len(y)), dtype=float)
    print("len(s), len(y):", len(s), len(y), "| NaN en s:", int(np.isnan(s).sum()),
          "| valores unicos (primeros 5):", np.unique(s)[:5])
    n = min(len(s), len(y))
    print("corr(s, y):", round(float(np.corrcoef(s[:n], y[:n])[0, 1]), 4),
          "| corr(s, indice):", round(float(np.corrcoef(s[:n], np.arange(n))[0, 1]), 4))
    print("sklearn sobre s_lote:", round(roc_auc_score(y[:n], s[:n]), 4))
    print("funcion AUC del lote:", round(float(auc_del_lote(y[:n], s[:n])), 4))
    print("valor guardado en CSV:", auc_csv)

    print("\nLECTURA")
    print("- standalone ~0.5 y sklearn(s_lote) ~0.01 -> bug al construir/alinear scores o etiquetas")
    print("- standalone ~0.5, sklearn(s_lote) ~0.5, funcion del lote ~0.01 -> bug en la funcion de AUC")
    print("- standalone ya ~0.01 -> etiquetas invertidas o desalineadas en la carga (afecta a los 3 brazos)")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))