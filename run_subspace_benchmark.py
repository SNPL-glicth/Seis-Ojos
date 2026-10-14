"""Benchmark TSB-AD Tuning-43 para SubspaceResidualDetector (gating causal)."""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from sixeyes.evaluation.tsb_runner import run_single_series
from sixeyes.detectors.subespacio import SubspaceResidualDetector

LIMITE_S = 120.0


def resolve_tsb_paths() -> tuple[Path, Path]:
    candidates = [
        ROOT_DIR / "benchmarks" / "tsb_ad" / "TSB-AD",
        ROOT_DIR.parent / "benchmarks" / "tsb_ad" / "TSB-AD",
    ]
    for base in candidates:
        lf = base / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv"
        dd = base / "Datasets" / "TSB-AD-U"
        if lf.is_file() and dd.is_dir():
            return lf, dd
    # Fallback portable: deja que el error sea explícito.
    base = candidates[0]
    return (base / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv",
            base / "Datasets" / "TSB-AD-U")


def main():
    start_global = time.time()
    list_file, data_dir = resolve_tsb_paths()

    df_list = pd.read_csv(list_file)
    files_to_run = df_list[~df_list['file_name'].str.contains('_NAB_')]['file_name'].tolist()
    print("Iniciando evaluación de SubspaceResidualDetector (gating causal) en Tuning...")
    print(f"Total series a evaluar: {len(files_to_run)}")

    det_name = "Subspace_Gated"

    def factory():
        # Experimento histórico con gating 0.70 explícito (default oficial: None).
        return SubspaceResidualDetector(gate_threshold=0.70)

    rows = []
    not_evaluated = []

    for i, f in enumerate(files_to_run):
        dataset = f.split('_')[1]
        res = run_single_series(factory, data_dir / f, timeout_s=None)

        if res["status"] == "OK":
            m = res["metrics"]
            total = res["time_detector"] + res["time_metrics"]
            rows.append({
                "archivo": f, "dataset": dataset, "largo": res["length"],
                "VUS-PR": m.get("VUS-PR"), "AUC-PR": m.get("AUC-PR"),
                "AUC-ROC": m.get("AUC-ROC"),
                "segundos_detector": res["time_detector"],
                "segundos_metrics": res["time_metrics"],
                "supera_120s": total > LIMITE_S,
            })
            print(f"[{i+1:02d}/{len(files_to_run)}] {dataset:<10} | VUS-PR: {m.get('VUS-PR', 0.0):.4f} | AUC-ROC: {m.get('AUC-ROC', 0.0):.4f} | {f} ({res['time_detector']:.2f}s det, {res['time_metrics']:.2f}s met)")
        else:
            not_evaluated.append((f, res["error"]))
            rows.append({
                "archivo": f, "dataset": dataset, "largo": None,
                "VUS-PR": None, "AUC-PR": None, "AUC-ROC": None,
                "segundos_detector": None, "segundos_metrics": None,
                "supera_120s": res["time"] > LIMITE_S,
                "estado": "NO_EVALUADA: " + str(res["error"]),
            })
            print(f"[{i+1:02d}/{len(files_to_run)}] {dataset:<10} | NO_EVALUADA: {f} ({res['error']})")

    df_raw = pd.DataFrame(rows)
    # PROHIBIDO sobrescribir el baseline: solo archivo separado + results/.
    df_raw.to_csv(ROOT_DIR / f"{det_name}_tsb_results.csv", index=False)
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(results_dir / f"{det_name}_tsb_results.csv", index=False)

    ok = df_raw.dropna(subset=["VUS-PR"])
    print(f"\nEvaluadas: {len(ok)}/{len(files_to_run)} | Fallidas: {len(not_evaluated)}")

    if not ok.empty:
        print("\n" + "=" * 70)
        print("REPORTE: SUBESPACIO GATED (baseline inmutable, no sobrescrito)")
        print("=" * 70)
        baseline_path = results_dir / "Subspace_tsb_results.csv"
        if baseline_path.is_file():
            df_sub_base = pd.read_csv(baseline_path).dropna(subset=["VUS-PR"])
            print(f"VUS-PR Promedio por Serie:")
            print(f"  Subspace Base (inmutable):  {df_sub_base['VUS-PR'].mean():.4f}")
            print(f"  Subspace (Gated, actual):   {ok['VUS-PR'].mean():.4f} (Diff = {ok['VUS-PR'].mean() - df_sub_base['VUS-PR'].mean():+.4f})")
        else:
            print("Baseline Subspace_tsb_results.csv no encontrado; se omite comparativa.")
            print(f"  Subspace (Gated, actual): {ok['VUS-PR'].mean():.4f}")

        total_det = ok['segundos_detector'].sum()
        total_met = ok['segundos_metrics'].sum()
        print(f"\nTiempos de Ejecución:")
        print(f"  Tiempo total de detector: {total_det:.2f} s ({total_det/len(ok):.3f}s / serie)")
        print(f"  Tiempo total de métricas: {total_met:.2f} s")
        print(f"  Tiempo global acumulado:  {time.time() - start_global:.2f} s")


if __name__ == "__main__":
    main()
