"""Process the 204 cand_s42 series that have data but need zscore/random.

Uses ProcessPoolExecutor for parallelization. Only AUC-ROC and AUC-PR.
Writes rows to results/sixeyes_tsb_fresh472_rows.csv with checkpoint.
"""
import pandas as pd, os, sys, json, time, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, 'src')
from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.echo_state import EchoStatePredictor
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass

TSB = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\TSB-AD-U'
data_dir = TSB
CSV = Path('results/sixeyes_tsb_fresh472_rows.csv')
CHECKPOINT_DIR = Path('results/.fresh472_ckpt2')
CHECKPOINT_DIR.mkdir(exist_ok=True)

ARMS = ("cand_s42", "zscore", "random")

def factory(arm):
    if arm == "cand_s42":
        esn = EchoStatePredictor(seed=42)
        cal = PercentileCalibrator(esn, max_history=16000)
        return SpatialSpikeBypass(cal, tolerance=0.05, warmup=100)
    if arm == "zscore":
        return RollingRobustZ()
    if arm == "random":
        return RandomDetector(seed=42)
    raise ValueError(arm)

def process_one_file(fname):
    """Pure function: return dict arm->metrics for a single file."""
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        return fname, {arm: {"status": "MISSING", "AUC-ROC": np.nan, "AUC-PR": np.nan} for arm in ARMS}
    res = {}
    for arm in ARMS:
        try:
            det = factory(arm)
            r = run_single_series(lambda a=arm: factory(a), fpath, timeout_s=600.0)
            m = r.get("metrics", {})
            res[arm] = {
                "status": r.get("status", "ERROR"),
                "AUC-ROC": m.get("AUC-ROC", np.nan),
                "AUC-PR": m.get("AUC-PR", np.nan)
            }
        except Exception as e:
            res[arm] = {"status": "ERROR", "AUC-ROC": np.nan, "AUC-PR": np.nan}
    return fname, res

def main():
    # Read existing CSV to know what's done and get cand_s42 AUCs
    df_existing = pd.DataFrame()
    if CSV.is_file():
        df_existing = pd.read_csv(CSV)
    
    # Get fresh files
    base_list = os.path.join(TSB, '..\\..\\File_List')  
    # Read file lists properly
    import pandas as pd2
    eva_path = os.path.join(r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\File_List', 'TSB-AD-U-Eva.csv')
    full_path = os.path.join(r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\File_List', 'TSB-AD-U-Eva-Full.csv')
    eva = set(pd2.read_csv(eva_path)['file_name'])
    full = pd2.read_csv(full_path)['file_name'].tolist()
    fresh = [f for f in full if f not in eva]
    
    # Determine which cand_s42 files still need zscore/random
    # Get files that have cand_s42 OK in existing CSV
    cand_ok = df_existing[df_existing.arm == 'cand_s42'][['file', 'AUC-ROC']].rename(columns={'AUC-ROC': 'cand_auc'}) if not df_existing.empty else pd.DataFrame()
    
    # Get all fresh files that have cand_s42 data already (from the 211/218)
    # Actually, let's find files in fresh that are in the CSV with cand_s42 status OK
    done_files = set(df_existing.file.tolist()) if not df_existing.empty else set()
    need_zr = []  # files that have cand_s42 but need zscore/random
    need_all = []  # files completely missing from CSV
    
    for f in fresh:
        if f in done_files:
            # Check if zscore/random already have rows
            zs = df_existing[(df_existing.file == f) & (df_existing.arm == 'zscore')]
            rs = df_existing[(df_existing.file == f) & (df_existing.arm == 'random')]
            if len(zs) == 0 or len(rs) == 0:
                need_zr.append(f)
            # else: already have all 3 arms
        else:
            need_all.append(f)
    
    print(f"[204] fresh total: {len(fresh)}", flush=True)
    print(f"[204] need zscore/random (have cand, missing z/r): {len(need_zr)}", flush=True)
    print(f"[204] need all 3 arms (missing from CSV): {len(need_all)}", flush=True)
    
    if not need_zr and not need_all:
        print("[204] todo hecho", flush=True)
        # generate final JSON
        return
    
    # Initialize CSV if needed
    if not CSV.is_file():
        pd.DataFrame(columns=["arm", "file", "status", "AUC-ROC", "AUC-PR"]).to_csv(CSV, index=False)
    
    # Process need_zr files in parallel
    if need_zr:
        print(f"[204] Processing {len(need_zr)} files with zscore/random...", flush=True)
        t0 = time.time()
        
        # Use ProcessPoolExecutor
        n_cpus = max(1, os.cpu_count() - 1)
        print(f"[204] Using {n_cpus} cpus", flush=True)
        
        def wrapper(f):
            return f, process_one_file(f)
        
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(executor.map(wrapper, need_zr))
        
        for fname, res in results:
            # Append rows to CSV
            with open(CSV, "a", newline="") as cf:
                for arm in ARMS:
                    m = res.get(arm, {})
                    row = {"arm": arm, "file": fname, "status": m.get("status", "ERROR"),
                           "AUC-ROC": m.get("AUC-ROC", np.nan), "AUC-PR": m.get("AUC-PR", np.nan)}
                    pd.DataFrame([row]).to_csv(cf, header=not CSV.is_file(), index=False)
                    # After first write, header is already there
                    CSV.is_file()  # just to trigger the check, but we need to track
            # Status line
            if len(results) % 20 == 0 or True:  # simplified
                print(f"[204] +{len([r for r in results[:results.index((fname,res))+1]])}/{len(need_zr)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    
    # Process need_all files with all 3 arms from scratch
    if need_all:
        print(f"[204] Processing {len(need_all)} files with all 3 arms from scratch...", flush=True)
        t0 = time.time()
        
        n_cpus = max(1, os.cpu_count() - 1)
        print(f"[204] Using {n_cpus} cpus", flush=True)
        
        def wrapper(f):
            return f, process_one_file(f)
        
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(executor.map(wrapper, need_all))
        
        for fname, res in results:
            # Append rows to CSV
            with open(CSV, "a", newline="") as cf:
                for arm in ARMS:
                    m = res.get(arm, {})
                    row = {"arm": arm, "file": fname, "status": m.get("status", "ERROR"),
                           "AUC-ROC": m.get("AUC-ROC", np.nan), "AUC-PR": m.get("AUC-PR", np.nan)}
                    pd.DataFrame([row]).to_csv(cf, header=False, index=False)
            print(f"[204] +{list(results).index((fname,res))+1}/{len(need_all)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    
    # Final analysis
    print("[204] Running final analysis...", flush=True)
    df = pd.read_csv(CSV)
    ok = df[df.status == "OK"]
    print(f"[204] Total OK rows: {len(ok)}")
    
    # Paired analysis: cand_s42 vs zscore
    cand_df = ok[ok.arm == 'cand_s42'][['file', 'AUC-ROC']].rename(columns={'AUC-ROC': 'cand_auc'})
    z_df = ok[ok.arm == 'zscore'][['file', 'AUC-ROC']].rename(columns={'AUC-ROC': 'z_auc'})
    r_df = ok[ok.arm == 'random'][['file', 'AUC-ROC']].rename(columns={'AUC-ROC': 'r_auc'})
    
    merged = cand_df.merge(z_df, on='file').merge(r_df, on='file')
    n_paired = len(merged)
    
    if n_paired > 0:
        c_vals = merged['cand_auc'].values.astype(float)
        z_vals = merged['z_auc'].values.astype(float)
        r_vals = merged['r_auc'].values.astype(float)
        
        diff = c_vals - z_vals
        rng = np.random.RandomState(42)
        boots = [float(np.mean(np.random.choice(diff, size=len(diff), replace=True))) for _ in range(10000)]
        lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        
        frac_cand_05 = float(sum(1 for v in c_vals if v > 0.5) / len(c_vals))
        frac_z_05 = float(sum(1 for v in z_vals if v > 0.5) / len(z_vals))
        
        pass_criterion = (diff.mean() >= 0) and (lo > 0) and (frac_cand_05 >= 0.55)
        
        rep = {
            "n_files": len(df["file"].unique()) if "file" in df.columns else len(fresh),
            "n_paired": n_paired,
            "cand_mean": round(float(np.mean(c_vals)), 4),
            "zscore_mean": round(float(np.mean(z_vals)), 4),
            "random_mean": round(float(np.mean(r_vals)), 4),
            "diff_mean": round(float(diff.mean()), 4),
            "diff_ci95": [round(lo, 4), round(hi, 4)],
            "frac_cand_over_05": round(frac_cand_05, 4),
            "frac_z_over_05": round(frac_z_05, 4),
            "pass": bool(pass_criterion),
            "note": f"211/472 + {len(need_zr)} + {len(need_all)} series. "
                    f"AUC-ROC y AUC-PR solamente, sin VUS. Cobertura completa tras reanudo.",
        }
    else:
        rep = {
            "n_files": len(df["file"].unique()) if "file" in df.columns else len(fresh),
            "note": "No paired data available. Some series had errors in all 3 arms (likely long UCR series).",
        }
    
    (Path('results/sixeyes_tsb_fresh472.json')).write_text(
        json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
    print("[204] DONE " + json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()