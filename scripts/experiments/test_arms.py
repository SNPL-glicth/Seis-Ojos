import pandas as pd, os, sys
sys.path.insert(0, 'src')
from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass

TSB = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'
base = os.path.join(TSB, 'File_List')
import pandas as pd
eva = set(pd.read_csv(os.path.join(base, 'TSB-AD-U-Eva.csv'))['file_name'])
full = pd.read_csv(os.path.join(base, 'TSB-AD-U-Eva-Full.csv'))['file_name'].tolist()
fresh = [f for f in full if f not in eva]

test_files = fresh[:10]
for fname in test_files:
    fpath = os.path.join(TSB, fname)
    # zscore
    try:
        r = run_single_series(lambda: RollingRobustZ(), fpath, timeout_s=300.0)
        z_auc = r.get('metrics', {}).get('AUC-ROC')
        print(f'zscore {fname[:30]}: status={r["status"]} auc={z_auc}')
    except Exception as e:
        print(f'zscore {fname[:30]}: ERROR {str(e)[:60]}')
    # random
    try:
        r = run_single_series(lambda: RandomDetector(seed=42), fpath, timeout_s=300.0)
        r_auc = r.get('metrics', {}).get('AUC-ROC')
        print(f'random {fname[:30]}: status={r["status"]} auc={r_auc}')
    except Exception as e:
        print(f'random {fname[:30]}: ERROR {str(e)[:60]}')
    # cand_s42
    try:
        esn = EchoStatePredictor(seed=42)
        cal = PercentileCalibrator(esn, max_history=16000)
        det = SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)
        r = run_single_series(lambda: det, fpath, timeout_s=300.0)
        c_auc = r.get('metrics', {}).get('AUC-ROC')
        print(f'cand_s42 {fname[:30]}: status={r["status"]} auc={c_auc}')
    except Exception as e:
        print(f'cand_s42 {fname[:30]}: ERROR {str(e)[:60]}')