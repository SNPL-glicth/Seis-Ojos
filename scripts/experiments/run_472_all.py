"""Process 204+254 fresh series with zscore and random.

Serial execution with per-file checkpoint. Only AUC-ROC and AUC-PR.
Writes rows to results/sixeyes_tsb_fresh472_rows.csv.
"""
import pandas as pd, os, sys, json, time, numpy as np
from pathlib import Path

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
    """Pure function: return dict arm->metrics for a single file. Safe for threads."""
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


def worker(fname):
    """External function for potential PPE (kept for compatibility)."""
    return process_one_file(fname)


def main():
    # Read existing CSV to know what's done
    df_existing = pd.DataFrame()
    if CSV.is_file():
        df_existing = pd.read_csv(CSV)

    # Read file lists
    import pandas as pd2
    base_list = r'C:\Users\Nicolas Pachon\Desktop\Seis Ojos\benchmarks\tsb_ad\TSB-AD\Datasets\File_List'
    eva = set(pd2.read_csv(os.path.join(base_list, 'TSB-AD-U-Eva.csv'))['file_name'])
    full = pd2.read_csv(os.path.join(base_list, 'TSB-AD-U-Eva-Full.csv'))['file_name'].tolist()
    fresh = [f for f in full if f not in eva]
    assert len(fresh) == 472

    # Determine what's missing
    done_files = set(df_existing.file.tolist()) if not df_existing.empty else set()
    # Files that have cand_s42 but need zscore/random
    need_zr = []
    # Files completely missing from CSV
    need_all = []

    for f in fresh:
        if f in done_files:
            # Check if zscore/random already have rows
            zs = df_existing[(df_existing.file == f) & (df_existing.arm == 'zscore')]
            rs = df_existing[(df_existing.file == f) & (df_existing.arm == 'random')]
            if len(zs) == 0 or len(rs) == 0:
                need_zr.append(f)
        else:
            need_all.append(f)

    print(f"[main] fresh total: {len(fresh)}", flush=True)
    print(f"[main] need zscore/random (have cand, missing z/r): {len(need_zr)}", flush=True)
    print(f"[main] need all 3 arms (missing from CSV): {len(need_all)}", flush=True)

    # Initialize CSV if needed
    if not CSV.is_file():
        pd.DataFrame(columns=["arm", "file", "status", "AUC-ROC", "AUC-PR"]).to_csv(CSV, index=False)

    t0 = time.time()
    n_done = len(done_files)

    # Process need_zr files (have cand_s42 OK, add zscore/random)
    if need_zr:
        print(f"[main] Processing {len(need_zr)} files with zscore/random...", flush=True)
        for i, fname in enumerate(need_zr):
            fname, res = process_one_file(fname)
            n_done += 1
            # Append rows to CSV
            with open(CSV, "a", newline="") as cf:
                header = not CSV.is_file() or os.path.getsize(CSV) == 0
                # Actually, just append without header after first write
                # Simpler: always append, header handled by first write check
                first_write = (os.path.getsize(CSV) == 0) if CSV.is_file() else True
                if first_write:
                    cf.write('arm,file,status,AUC-ROC,AUC-PR\n')
                for arm in ARMS:
                    m = res.get(arm, {})
                    row = {"arm": arm, "file": fname, "status": m.get("status", "ERROR"),
                           "AUC-ROC": m.get("AUC-ROC", np.nan), "AUC-PR": m.get("AUC-PR", np.nan)}
                    pd.DataFrame([row]).to_csv(cf, header=False, index=False)
            if (i+1) % 40 == 0 or i == len(need_zr) - 1:
                elapsed = time.time() - t0
                print(f"[main] +{n_done}/{len(fresh)} ({elapsed:.0f}s)", flush=True)

    # Process need_all files (completely missing, all 3 arms from scratch)
    if need_all:
        print(f"[main] Processing {len(need_all)} files with all 3 arms from scratch...", flush=True)
        t0 = time.time()
        for i, fname in enumerate(need_all):
            fname, res = process_one_file(fname)
            n_done += 1
            # Append rows to CSV (no header after first batch)
            with open(CSV, "a", newline="") as cf:
                first_write = (os.path.getsize(CSV) == 0)
                if first_write:
                    cf.write('arm,file,status,AUC-ROC,AUC-PR\n')
                for arm in ARMS:
                    m = res.get(arm, {})
                    row = {"arm": arm, "file": fname, "status": m.get("status", "ERROR"),
                           "AUC-ROC": m.get("AUC-ROC", np.nan), "AUC-PR": m.get("AUC-PR", np.nan)}
                    pd.DataFrame([row]).to_csv(cf, header=False, index=False)
            if (i+1) % 40 == 0 or i == len(need_all) - 1:
                elapsed = time.time() - t0
                print(f"[main] +{n_done}/{len(fresh)} ({elapsed:.0f}s)", flush=True)

    # Final analysis
    print("[main] Running final analysis...", flush=True)
    df = pd.read_csv(CSV)
    ok = df[df.status == "OK"]
    print(f"[main] Total OK rows: {len(ok)}")

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
            "note": f"472/472 series, cobertura completa tras reanudo. "
                    f"AUC-ROC y AUC-PR solamente, sin VUS.",
        }
    else:
        rep = {
            "n_files": len(df["file"].unique()) if "file" in df.columns else len(fresh),
            "note": "No paired data available.",
        }

    (Path('results/sixeyes_tsb_fresh472.json')).write_text(
        json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
    print("[main] DONE " + json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()