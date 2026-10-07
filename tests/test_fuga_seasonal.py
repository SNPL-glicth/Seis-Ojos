import os
import pandas as pd
from detectors.seasonal import DailySeasonalRobustZ

def test_anti_fuga_seasonal_permanente():
    """
    e) Anti-fuga: alterar valores posteriores a k (k mucho mayor que el
    calentamiento) no cambia los scores hasta k, y hay scores > 0.0 antes de k.
    """
    # Usamos un archivo de datos real de NAB como en el diagnostic_6.py
    # La ruta asume que los tests corren desde la raíz del proyecto Six Eyes
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "nab", "data", "realAWSCloudwatch", "ec2_cpu_utilization_77c1ca.csv")
    
    if not os.path.exists(data_path):
        # Si no está el archivo, saltamos el test
        return
        
    df = pd.read_csv(data_path)
    # Reemplazamos los espacios por 'T' para simular lo que espera _parse_and_bucket
    timestamps = df["timestamp"].astype(str).str.replace(" ", "T").tolist()
    values = df["value"].tolist()
    
    k = len(values) // 2
    
    det_normal = DailySeasonalRobustZ()
    scores_normal = []
    for val, ts in zip(values, timestamps):
        scores_normal.append(det_normal.update(val, ts))
        
    det_alt = DailySeasonalRobustZ()
    scores_alt = []
    for i, (val, ts) in enumerate(zip(values, timestamps)):
        if i >= k:
            val = 9999.0
        scores_alt.append(det_alt.update(val, ts))
        
    # Check that there are scores > 0.0 before k
    non_zero = sum(1 for s in scores_normal[:k] if s > 0.0)
    assert non_zero > 0, "No hubo scores > 0.0 antes de k (test vacío)"
    
    # Check that the scores up to k are perfectly identical
    assert scores_normal[:k] == scores_alt[:k], "Hubo fuga: scores previos a k cambiaron al alterar el futuro"
