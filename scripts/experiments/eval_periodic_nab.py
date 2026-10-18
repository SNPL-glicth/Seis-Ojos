"""Script de evaluación oficial y auditoría para la rama periódica (PeriodicResidualDetector).

Evalúa el impacto de la rama periódica sobre las 58 series completas de NAB:
1. Medición de compuerta abierta (%) y período dominante por serie.
2. Comparación punto a punto con el baseline titular oficial (seed 42):
   - Con umbral oficial fijado (in-sample baseline: 0.99077).
   - Con barrido y re-optimización completa de NAB.
3. Métricas clave de parada:
   - Ventanas rescatadas de la brecha vs perdidas del baseline.
   - Delta H, Delta F_eff y cociente Delta F_eff / Delta H (referencia de equilibrio NAB ~ 17.01).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
GATES = ROOT / "archive" / "gates"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(GATES))

from exp_replay_pareado import load_all
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.peak_decay import StreamingPeakDecay
from sixeyes.detectors.spatial_spike_bypass import SpatialSpikeBypass
from sixeyes.detectors.refractory_suppressor import RefractorySuppressor
from sixeyes.experimental.periodic_residual import PeriodicResidualDetector, MaxFusion
from sixeyes.evaluation.nab_runner import NABRunner

NAB_PATH = ROOT.parent / "benchmarks" / "nab"
BASELINE_CSV = ROOT / "results" / "sixeyes_refractory_w80_t990_standard_scores.csv"
BASELINE_JSON = ROOT / "results" / "sixeyes_refractory_w80_t990.json"

K_WINDOW = 24
TAU = 2.0
CAL_W = 16000
SPATIAL_TOL = 0.05
SPATIAL_WARMUP = 100
REFR_W = 80
REFR_T = 0.990
DET_NAME = "sixeyes_periodic_fusion_s42"


class SeqPass:
    def __init__(self, seq):
        self.seq = [float(x) for x in seq]
        self.i = 0

    def update(self, value: float, timestamp: str = "") -> float:
        r = self.seq[self.i]
        self.i += 1
        return float(r)

    def reset(self):
        self.i = 0


def run_evaluation(seed: int = 42):
    print("=" * 70)
    print("EVALUACIÓN OFICIAL: PeriodicResidualDetector + ESN MaxFusion")
    print("=" * 70)

    series_data = load_all()
    print(f"Cargadas {len(series_data)} series desde raw_esn.")

    det_dir = NAB_PATH / "results" / DET_NAME
    det_dir.mkdir(parents=True, exist_ok=True)

    col = f"s{seed}"
    gate_stats = []

    print("\nGenerando detecciones causales punto a punto...")
    for rel, data in series_data.items():
        vals = data["value"]
        ts = data["timestamp"]
        raws_esn = data[col]

        # 1. Rama ESN (reutiliza crudos para idéntica causalidad)
        cal_esn = PercentileCalibrator(SeqPass(raws_esn), max_history=CAL_W)

        # 2. Rama Periódica
        per = PeriodicResidualDetector()
        cal_per = PercentileCalibrator(per, max_history=CAL_W)

        # Registro de compuerta abierta
        open_count = 0
        scores_per_raw = []
        for v in vals:
            s_raw = per.update(v)
            scores_per_raw.append(s_raw)
            if s_raw > 0.0:
                open_count += 1

        per_open_pct = (open_count / len(vals)) * 100.0 if len(vals) > 0 else 0.0
        gate_stats.append({
            "File": rel,
            "Length": len(vals),
            "GateOpenPct": per_open_pct,
        })

        # 3. Fusión MaxFusion aguas abajo
        # Reset de per para procesar en streaming dentro de la cadena
        per.reset()
        cal_esn.reset()
        cal_per.reset()

        fus = MaxFusion([cal_esn, cal_per])
        dec = StreamingPeakDecay(fus, k_window=K_WINDOW, tau=TAU)
        spat = SpatialSpikeBypass(dec, tolerance=SPATIAL_TOL, warmup=SPATIAL_WARMUP)
        refr = RefractorySuppressor(spat, w_rest=REFR_W, trigger_threshold=REFR_T)

        final_scores = [refr.update(v, t_str) for v, t_str in zip(vals, ts)]

        cat, fn = rel.split("/", 1)
        out_sub = det_dir / cat
        out_sub.mkdir(parents=True, exist_ok=True)
        out_csv = out_sub / f"{DET_NAME}_{fn}"
        df = pd.DataFrame({
            "timestamp": ts,
            "value": vals,
            "anomaly_score": final_scores,
            "label": np.zeros(len(final_scores), dtype=int),
        })
        df.to_csv(out_csv, index=False)

    df_gates = pd.DataFrame(gate_stats)
    print(f"\nResumen de actividad de compuerta en las {len(df_gates)} series:")
    print(f" - Series con compuerta abierta (>0% del tiempo): {(df_gates['GateOpenPct'] > 0).sum()} / {len(df_gates)}")
    print(f" - Media global de tiempo abierta: {df_gates['GateOpenPct'].mean():.2f}%")
    print("\nTop 5 series con mayor apertura:")
    print(df_gates.sort_values(by="GateOpenPct", ascending=False).head(5).to_string(index=False))

    print("\nEjecutando optimizador y scorer oficial de NAB...")
    runner = NABRunner(nab_root=NAB_PATH)
    scores = runner.ejecutar_scorer_oficial(DET_NAME)

    res_json = ROOT / "results" / f"{DET_NAME}.json"
    res_json.write_text(json.dumps({
        "detector": DET_NAME,
        "seed": seed,
        "scores": scores,
        "gate_stats": gate_stats
    }, indent=2), encoding="utf-8")

    # Copiar CSV de scores standard a results
    import shutil
    shutil.copy(det_dir / f"{DET_NAME}_standard_scores.csv",
                ROOT / "results" / f"{DET_NAME}_standard_scores.csv")

    print("\n" + "=" * 70)
    print("RESULTADOS SCORE NAB (NUEVA FUSIÓN PERIÓDICA):")
    for k, v in scores.items():
        print(f"   {k:25s} : {v:.2f}")
    print("=" * 70)

    # Comparación analítica contra baseline
    base_csv = ROOT / "results" / f"sixeyes_refractory_w80_t990_standard_scores.csv" if seed == 42 else ROOT / "results" / f"sixeyes_refractory_seed{seed}_w80_t990_standard_scores.csv"
    if not base_csv.is_file():
        base_csv = ROOT / "results" / f"sixeyes_refractory_w80_t990_standard_scores.csv"

    if base_csv.is_file():
        df_base = pd.read_csv(base_csv)
        df_new = pd.read_csv(ROOT / "results" / f"{DET_NAME}_standard_scores.csv")

        df_b_files = df_base.dropna(subset=["File"])
        df_n_files = df_new.dropna(subset=["File"])

        m = pd.merge(df_b_files, df_n_files, on="File", suffixes=("_base", "_fusion"))
        m["is_art"] = m["File"].str.startswith("artificial")

        # Desglose global
        tp_base = m["TP_base"].sum()
        tp_new = m["TP_fusion"].sum()
        fp_base = m["FP_base"].sum()
        fp_new = m["FP_fusion"].sum()

        delta_h = tp_new - tp_base
        delta_fp = fp_new - fp_base
        ratio = (delta_fp / delta_h) if delta_h != 0 else float("inf")

        print("\nAUDITORÍA Y DESCOMPOSICIÓN ANALÍTICA vs BASELINE:")
        print(f" - Ventanas Detectadas (H): {tp_base} -> {tp_new} (Delta H = {delta_h:+d})")
        print(f" - Falsas Alarmas (FP):     {fp_base} -> {fp_new} (Delta FP = {delta_fp:+d})")
        print(f" - Cociente marginal medido: {ratio:.2f} FPs por ventana (Equilibrio teórico ~ 17.01)")

        # Desglose Sintéticas vs Reales
        art = m[m["is_art"]]
        real = m[~m["is_art"]]

        d_h_art = art["TP_fusion"].sum() - art["TP_base"].sum()
        d_fp_art = art["FP_fusion"].sum() - art["FP_base"].sum()
        d_score_art = art["Score_fusion"].sum() - art["Score_base"].sum()

        d_h_real = real["TP_fusion"].sum() - real["TP_base"].sum()
        d_fp_real = real["FP_fusion"].sum() - real["FP_base"].sum()
        d_score_real = real["Score_fusion"].sum() - real["Score_base"].sum()

        print("\nDESGLOSE POR ORIGEN DE DATOS:")
        print(f" * SINTÉTICAS (11 series): Delta H = {d_h_art:+d}, Delta FP = {d_fp_art:+d}, Delta Score Raw = {d_score_art:+.3f}")
        print(f" * REALES (47 series):     Delta H = {d_h_real:+d}, Delta FP = {d_fp_real:+d}, Delta Score Raw = {d_score_real:+.3f}")

        ganadas = m[m["TP_fusion"] > m["TP_base"]]
        perdidas = m[m["TP_fusion"] < m["TP_base"]]

        print(f"\nVentanas recuperadas (+): {len(ganadas)}")
        if len(ganadas) > 0:
            print(ganadas[["File", "TP_base", "TP_fusion", "FP_base", "FP_fusion"]].to_string(index=False))

        print(f"\nVentanas perdidas por supresión (-): {len(perdidas)}")
        if len(perdidas) > 0:
            print(perdidas[["File", "TP_base", "TP_fusion", "FP_base", "FP_fusion"]].to_string(index=False))

    return scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    DET_NAME = f"sixeyes_periodic_fusion_s{args.seed}"
    run_evaluation(seed=args.seed)
