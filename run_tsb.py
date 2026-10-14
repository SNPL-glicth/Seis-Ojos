"""Benchmark TSB-AD Tuning-43 para el ensamble Combinado (experimental)."""

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
from sixeyes.detectors.combinador import Combinado

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
    base = candidates[0]
    return (base / "Datasets" / "File_List" / "TSB-AD-U-Tuning.csv",
            base / "Datasets" / "TSB-AD-U")


def main():
    start_global = time.time()
    list_file, data_dir = resolve_tsb_paths()

    df_list = pd.read_csv(list_file)
    files_to_run = df_list[~df_list['file_name'].str.contains('_NAB_')]['file_name'].tolist()
    print(f"Series en Tuning (excluyendo NAB): {len(files_to_run)}")

    det_name = "Combinado"

    def factory():
        return Combinado()

    rows = []
    not_evaluated = []

    print("\nIniciando evaluación secuencial...")
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
            print(f"[{i+1}/{len(files_to_run)}] {f}: OK (det: {res['time_detector']:.2f}s, met: {res['time_metrics']:.2f}s)")
        else:
            not_evaluated.append((f, res["error"]))
            rows.append({
                "archivo": f, "dataset": dataset, "largo": None,
                "VUS-PR": None, "AUC-PR": None, "AUC-ROC": None,
                "segundos_detector": None, "segundos_metrics": None,
                "supera_120s": res["time"] > LIMITE_S,
                "estado": "NO_EVALUADA: " + str(res["error"]),
            })
            print(f"[{i+1}/{len(files_to_run)}] {f}: NO_EVALUADA ({res['error']})")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(ROOT_DIR / f"{det_name}_tsb_results.csv", index=False)
    ok = df_raw.dropna(subset=["VUS-PR"])

    print(f"\nEvaluadas: {len(ok)} | NO_EVALUADAS: {len(not_evaluated)}")
    for f, e in not_evaluated:
        print(f"  NO_EVALUADA {f}: {e}")

    if not ok.empty:
        print("\n--- REPORTES (cálculo propio) ---")
        print(f"{det_name} VUS-PR promedio por serie: {ok['VUS-PR'].mean():.4f}")
        print(f"{det_name} VUS-PR promedio por dataset: {ok.groupby('dataset')['VUS-PR'].mean().mean():.4f}")
        print(f"Tiempo total: {time.time() - start_global:.2f} s")


if __name__ == "__main__":
    main()
