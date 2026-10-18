import pandas as pd, os, numpy as np

df = pd.read_csv('results/sixeyes_tsb_fresh472_rows.csv')
ok = df[df.status=='OK']
cand_files = set(ok[ok.arm=='cand_s42']['file'].tolist())
all_ok_files = set(ok.file.tolist())

base = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'
full_filelist = [f for f in os.listdir(base) if not f.startswith('.')]

print('Total files in CSV:', len(all_ok_files))
print('cand_s42 files:', len(cand_files))
print('Fresh files on disk:', len(full_filelist))

missing = full_filelist - all_ok_files
print('Files on disk but NOT in CSV:', len(missing))
print('Sample missing:', list(missing)[:5])

# Check lengths for cand_s42 files
cand_lens = []
for f in cand_files:
    fpath = os.path.join(base, f)
    if os.path.exists(fpath):
        lab = pd.read_csv(fpath, usecols=['Label'])['Label'].values
        cand_lens.append((f, len(lab)))
cand_lens.sort(key=lambda x: x[1])
print('cand_s42 lengths range:', cand_lens[0][1], '-', cand_lens[-1][1])
print('cand_s42 median length:', np.median([x[1] for x in cand_lens]))

# missing lengths
miss_lens = []
for f in list(missing)[:30]:
    fpath = os.path.join(base, f)
    if os.path.exists(fpath):
        lab = pd.read_csv(fpath, usecols=['Label'])['Label'].values
        miss_lens.append((f, len(lab)))
miss_lens.sort(key=lambda x: x[1])
print('missing (sample) lengths range:', miss_lens[0][1] if miss_lens else 'N/A', '-', miss_lens[-1][1] if miss_lens else 'N/A')
if len(miss_lens) > 0:
    print('missing median length:', np.median([x[1] for x in miss_lens]))

# Check a few missing files with try/except on run_single_series
from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass

print('\\n=== Testing 4 missing files with cand_s42 detector ===')
test_missing = list(missing)[:4]
for f in test_missing:
    fpath = os.path.join(base, f)
    esn = EchoStatePredictor(seed=42)
    cal = PercentileCalibrator(esn, max_history=16000)
    det = SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)
    try:
        r = run_single_series(lambda: det, fpath, timeout_s=600.0)
        print(f'{f[:30]}: status={r.get(\"status\")} metrics keys={list(r.get(\"metrics\",{}).keys())} auc={r.get(\"metrics\",{}).get(\"AUC-ROC\")}')
    except Exception as e:
        print(f'{f[:30]}: ERROR {type(e).__name__}: {str(e)[:100]}')
" 2>&1