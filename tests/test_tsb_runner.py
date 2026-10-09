import os
import pytest
import numpy as np
import pandas as pd
from src.evaluation.tsb_runner import run_single_series, get_tuning_files, _get_tsb_metrics
from src.detectors.calibrador_percentil import PercentileCalibrator
from src.detectors.zscore_robusto import RollingRobustZ

def test_runner_descarta_nab_y_calcula_metricas(tmp_path):
    list_csv = tmp_path / "Tuning.csv"
    list_csv.write_text("file_name\n001_NAB_test.csv\n002_WSD_test.csv\n003_NAB_test2.csv\n004_SMD_test.csv\n")
    files = get_tuning_files(str(list_csv))
    assert files == ["002_WSD_test.csv", "004_SMD_test.csv"]

@pytest.mark.skipif(
    not os.path.exists(r"..\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U\001_NAB_id_1_Facility_tr_1007_1st_2014.csv"),
    reason="Datos no encontrados"
)
def test_cordura_metricas():
    file_path = r"..\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U\001_NAB_id_1_Facility_tr_1007_1st_2014.csv"
    
    # 1. scores = etiquetas -> esperado cercano a 1
    class PerfectDetector:
        def __init__(self):
            self.i = 0
            df = pd.read_csv(file_path).dropna()
            self.labels = df['Label'].astype(int).to_numpy()
        def update(self, val, idx):
            score = float(self.labels[self.i])
            self.i += 1
            return score
            
    res1 = run_single_series(PerfectDetector, file_path)
    assert res1["status"] == "OK"
    print(f"\n1. VUS-PR (scores=etiquetas): {res1['metrics']['VUS-PR']}")
    
    # 2. scores aleatorios (semilla fija)
    class RandomScores:
        def __init__(self):
            np.random.seed(42)
        def update(self, val, idx):
            return float(np.random.rand())
            
    res2 = run_single_series(RandomScores, file_path)
    assert res2["status"] == "OK"
    
    df = pd.read_csv(file_path).dropna()
    prop_1 = df['Label'].sum() / len(df)
    print(f"2. VUS-PR (aleatorio): {res2['metrics']['VUS-PR']}, Proporción etiqueta 1: {prop_1}")
    
    # 3. PercentileCalibrator(RollingRobustZ)
    res3 = run_single_series(lambda: PercentileCalibrator(RollingRobustZ()), file_path)
    assert res3["status"] == "OK"
    print(f"3. VUS-PR (Percentile+RobustZ): {res3['metrics']['VUS-PR']}")
