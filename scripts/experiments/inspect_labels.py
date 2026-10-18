import pandas as pd, os, numpy as np, sys
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

print('=== TOTAL FRESH: ===', len(fresh))

# Read the rows CSV to see which arms have data
df = pd.read_csv('results/sixeyes_tsb_fresh472_rows.csv')
ok = df[df.status=='OK']
arms_in_csv = ok.arm.unique().tolist()
print('Arms in CSV:', arms_in_csv)
print('cand_s42 rows:', len(ok[ok.arm=='cand_s42']))
print('zscore rows:', len(ok[ok.arm=='zscore']))
print('random rows:', len(ok[ok.arm=='random']))

# For each arm, get files and their label stats
for arm in ['cand_s42', 'zscore', 'random']:
    adf = ok[ok.arm==arm]
    print(f'\n--- {arm}: {len(adf)} files ---')
    # Get unique files
    files = adf.file.tolist()
    # For a sample, check label stats
    sample_files = files[:5]
    for f in sample_files:
        fpath = os.path.join(TSB, f)
        if not os.path.exists(fpath):
            print(f'  MISSING FILE: {f[:40]}')
            continue
        # Read the original CSV to get labels
        try:
            labels_df = pd.read_csv(fpath, usecols=['Label'])
            lab = labels_df['Label'].values
            print(f'  {f[:40]}... n={len(lab)} sum={lab.sum()} unique={np.unique(lab)[:5]}')
        except Exception as e:
            print(f'  {f[:40]}... ERROR reading Label: {str(e)[:50]}')
    # Also check all files for this arm quickly
    # Check if any file has all-0 or all-1 labels
    all_zero = 0
    all_one = 0
    total = 0
    for f in files:
        fpath = os.path.join(TSB, f)
        if not os.path.exists(fpath):
            continue
        try:
            labels_df = pd.read_csv(fpath, usecols=['Label'])
            lab = labels_df['Label'].values
            total += 1
            if np.all(lab == 0):
                all_zero += 1
            elif np.all(lab == 1):
                all_one += 1
        except:
            pass
    print(f'  Total files readable: {total}, all-zero: {all_zero}, all-one: {all_one}')

# Now check cand_s42 specifically: which files have data, and their label stats
print('\n=== CAND_S42 DETAIL ===')
cand_df = ok[ok.arm=='cand_s42']
cand_files = cand_df.file.tolist()
print(f'Files with cand_s42: {len(cand_files)}')
zero_count = 0
one_count = 0
other_count = 0
total_check = 0
for f in cand_files:
    fpath = os.path.join(TSB, f)
    if not os.path.exists(fpath):
        continue
    try:
        labels_df = pd.read_csv(fpath, usecols=['Label'])
        lab = labels_df['Label'].values
        total_check += 1
        if np.all(lab == 0):
            zero_count += 1
        elif np.all(lab == 1):
            one_count += 1
        else:
            other_count += 1
    except:
        pass
print(f'Total readable: {total_check}')
print(f'All-zero labels: {zero_count}')
print(f'All-one labels: {one_count}')
print(f'Has variation: {other_count}')

# Check zscore/random on the cand_s42 files - just AUC-ROC quick
print('\n=== QUICK ZSCORE/RANDOM on CAND_S42 FILES ===')
for arm_name, detector in [('zscore', RollingRobustZ()), ('random', RandomDetector(seed=42))]:
    arm_df = ok[ok.arm==arm_name] if arm_name in ok.arm.unique() else pd.DataFrame()
    files_a = arm_df.file.tolist() if len(arm_df) > 0 else []
    print(f'{arm_name}: {len(files_a)} files in CSV')
    # Check first 3 files
    for f in files_a[:3]:
        fpath = os.path.join(TSB, f)
        if not os.path.exists(fpath):
            print(f'  {f[:30]}: MISSING')
            continue
        try:
            r = run_single_series(lambda: detector, fpath, timeout_s=300.0)
            z_auc = r.get('metrics', {}).get('AUC-ROC')
            z_st = r.get('status')
            print(f'  {f[:30]}: status={z_st} auc={z_auc}')
        except Exception as e:
            print(f'  {f[:30]}: ERROR {str(e)[:40]}')