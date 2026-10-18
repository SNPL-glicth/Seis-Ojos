import pandas as pd, os, numpy as np, sys
sys.path.insert(0, 'src')
from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass

TSB = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'
base = os.path.join(TSB, 'TSB-AD-U')  # data is directly in TSB-AD-U? Let me check
# Actually from earlier: data is in TSB-AD-U subdir
data_dir = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'

df = pd.read_csv('results/sixeyes_tsb_fresh472_rows.csv')
ok = df[df.status=='OK']
cand_files = ok[ok.arm=='cand_s42']['file'].tolist()
print('cand_s42 files in CSV:', len(cand_files))

# Run zscore on each cand_s42 file
z_results = {}
r_results = {}

print('Running zscore on', len(cand_files), 'files...')
for f in cand_files:
    fpath = os.path.join(data_dir, f)
    if not os.path.exists(fpath):
        print(f'MISSING: {f[:30]}')
        continue
    try:
        r = run_single_series(lambda: RollingRobustZ(), fpath, timeout_s=600.0)
        z_auc = r.get('metrics', {}).get('AUC-ROC')
        z_st = r.get('status')
        z_results[f] = {'status': z_st, 'auc': z_auc}
        if z_st != 'OK':
            print(f'zscore {f[:30]}: status={z_st} auc={z_auc}')
    except Exception as e:
        print(f'zscore {f[:30]}: ERROR {str(e)[:80]}')
        z_results[f] = {'status': 'ERROR', 'auc': np.nan}

print('\\nRunning random on', len(cand_files), 'files...')
for f in cand_files:
    fpath = os.path.join(data_dir, f)
    if not os.path.exists(fpath):
        continue
    try:
        r = run_single_series(lambda: RandomDetector(seed=42), fpath, timeout_s=600.0)
        r_auc = r.get('metrics', {}).get('AUC-ROC')
        r_st = r.get('status')
        r_results[f] = {'status': r_st, 'auc': r_auc}
        if r_st != 'OK':
            print(f'random {f[:30]}: status={r_st} auc={r_auc}')
    except Exception as e:
        print(f'random {f[:30]}: ERROR {str(e)[:80]}')
        r_results[f] = {'status': 'ERROR', 'auc': np.nan}

# Build comparison dataframe
print('\\n=== Building comparison ===')
cand_aucs = []
z_aucs = []
r_aucs = []
files_ok = []

for f in cand_files:
    if f in z__results and z__results[f]['status'] == 'OK' and f in r_results and r_results[f]['status'] == 'OK':
        if not np.isnan(z__results[f]['auc']) and not np.isnan(r_results[f]['auc']):
            cand_aucs.append(z__results[f]['auc'] if False else None)  # cand already known
            # Actually we need cand AUC from the CSV
            break  # will restructure below

# Better: read cand AUCs from existing CSV
cand_auc_dict = {}
for _, row in ok[ok.arm=='cand_s42'].iterrows():
    cand_auc_dict[row['file']] = row['AUC-ROC']

# Now collect paired data
paired = []
for f in cand_files:
    c_auc = cand_auc_dict.get(f)
    if c_auc is None:
        continue
    z_auc = z__results.get(f, {}).get('auc')
    r_auc = r_results.get(f, {}).get('auc')
    z_st = z_results.get(f, {}).get('status')
    r_st = r_results.get(f, {}).get('status')
    if z_auc is not None and r_auc is not None and not np.isnan(z_auc) and not np.isnan(r_auc):
        paired.append({
            'file': f,
            'cand_auc': c_auc,
            'z_auc': z_auc,
            'r_auc': r_auc
        })

print(f'Paired files: {len(paired)} out of {len(cand_files)}')
if len(paired) > 0:
    cand_vals = [p['cand_auc'] for p in paired]
    z_vals = [p['z_auc'] for p in paired]
    r_vals = [p['r_auc'] for p in paired]
    
    import numpy as np
    from scipy import stats
    
    diff = np.array(cand_vals) - np.array(z_vals)
    # Bootstrap CI for mean difference
    rng = np.random.RandomState(42)
    boots = [float(np.mean(np.random.choice(diff, size=len(diff), replace=True))) for _ in range(10000)]
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    
    frac_over_05 = float(sum(1 for v in cand_vals if v > 0.5) / len(cand_vals))
    frac_z_over_05 = float(sum(1 for v in z_vals if v > 0.5) / len(z_vals))
    
    print(f'\\nResults ({len(paired)} paired series):')
    print(f'  cand mean AUC: {float(np.mean(cand_vals)):.4f}')
    print(f'  zscore mean AUC: {float(np.mean(z_vals)):.4f}')
    print(f'  random mean AUC: {float(np.mean(r_vals)):.4f}')
    print(f'  diff mean (cand - zscore): {float(diff.mean()):.4f}')
    print(f'  diff 95% CI bootstrap: [{lo:.4f}, {hi:.4f}]')
    print(f'  frac cand > 0.5: {frac_over_05:.4f}')
    print(f'  frac zscore > 0.5: {frac_z_over_05:.4f}')
    
    pass_criterion = (diff.mean() >= 0) and (lo > 0) and (frac_over_05 >= 0.55)
    print(f'  PASSES pre-registered criterion: {pass_criterion}')
    print(f'  (cand mean >= zscore mean with CI not crossing 0, AND >=55% series > 0.5)')
else:
    print('No paired data available - check why zscore/random failed')
" 2>&1