"""Script principal para ejecutar la evaluación de Six Eyes en el benchmark oficial NAB.

Pipeline oficial congelado:
    DailySeasonalRobustZ(15min) + PercentileCalibrator + StreamingPeakDecay

Uso estándar reproducible:
    python run_nab.py --detector peak_decay --k-window 160 --tau 1.5

Cada ejecución guarda su reporte aislado en ``results/sixeyes_<config>.json``.
Nunca se muta ni sobrescribe el ``final_results.json`` oficial de NAB.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
NAB_DIR = ROOT_DIR / "benchmarks" / "nab"
if not NAB_DIR.is_dir():
    NAB_DIR = ROOT_DIR.parent / "benchmarks" / "nab"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(NAB_DIR) not in sys.path:
    sys.path.insert(0, str(NAB_DIR))

from sixeyes.detectors.aleatorio import RandomDetector
from sixeyes.detectors.zscore_robusto import RollingRobustZ
from sixeyes.detectors.seasonal import DailySeasonalRobustZ
# Baselines legacy: importación directa de módulo (no del namespace oficial).
from sixeyes.detectors.biseasonal import BiSeasonalRobustZ
from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
from sixeyes.detectors.subespacio import SubspaceResidualDetector
from sixeyes.detectors.peak_decay import StreamingPeakDecay, DEFAULT_K_WINDOW, DEFAULT_TAU
from sixeyes.evaluation.nab_runner import NABRunner

RUTA_DEFECTO_NAB = NAB_DIR

# Única lista válida: coincide 1:1 con las ramas implementadas abajo.
DETECTORES_SOPORTADOS = [
    "peak_decay",
    "seasonal_pct",
    "seasonal",
    "biseasonal_pct",
    "biseasonal",
    "zscore_pct",
    "zscore",
    "subspace",
    "random",
]


def build_factory(detector: str, bucket_minutes: int, k_window: int, tau: float,
                  cal_window: int = 16000):
    """Fábrica única del pipeline oficial y baselines. Sin variantes fantasma.

    NOTA: el nombre del detector nunca contiene puntos (NAB parsea
    ``split('.')[0]`` y los puntos colapsan los 3 perfiles en una sola clave).
    ``tau=2.0`` se serializa como ``tau2p0``.
    ``cal_window<=0`` = calibrador no acotado (solo reproducción histórica).
    """
    tau_tag = str(tau).replace(".", "p")
    cal_tag = f"_w{cal_window}" if cal_window > 0 else "_wInf"
    max_hist = cal_window if cal_window > 0 else None
    if detector == "peak_decay":
        def factory(bm=bucket_minutes, kw=k_window, t=tau, mh=max_hist):
            return StreamingPeakDecay(
                PercentileCalibrator(DailySeasonalRobustZ(bucket_size_minutes=bm),
                                     max_history=mh),
                k_window=kw,
                tau=t,
            )
        name = f"sixeyes_peak_decay_b{bucket_minutes}_k{k_window}_tau{tau_tag}{cal_tag}"
        return factory, name
    if detector == "random":
        return (lambda: RandomDetector(seed=42)), "sixeyes_random"
    if detector == "zscore":
        return (lambda: RollingRobustZ()), "sixeyes_zscore"
    if detector == "zscore_pct":
        return (lambda: PercentileCalibrator(RollingRobustZ())), "sixeyes_zscore_pct"
    if detector == "biseasonal_pct":
        return (lambda: PercentileCalibrator(BiSeasonalRobustZ())), "sixeyes_biseasonal_pct"
    if detector == "biseasonal":
        return (lambda: BiSeasonalRobustZ()), "sixeyes_biseasonal"
    if detector == "seasonal_pct":
        return (lambda: PercentileCalibrator(DailySeasonalRobustZ())), "sixeyes_seasonal_pct"
    if detector == "seasonal":
        return (lambda: DailySeasonalRobustZ()), "sixeyes_seasonal"
    if detector == "subspace":
        return (lambda: SubspaceResidualDetector()), "sixeyes_subspace"
    raise ValueError(f"Detector desconocido: {detector}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de Six Eyes en NAB")
    parser.add_argument(
        "--detector",
        choices=DETECTORES_SOPORTADOS,
        default="peak_decay",
        help="Detector a evaluar (por defecto: 'peak_decay' oficial).",
    )
    parser.add_argument("--k-window", type=int, default=DEFAULT_K_WINDOW)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--bucket-minutes", type=int, default=15)
    parser.add_argument("--cal-window", type=int, default=16000,
                        help="Ventana ECDF del calibrador. <=0 = no acotado "
                             "(solo reproducción histórica K=160, coste O(N)).")
    parser.add_argument(
        "--nab-path", type=Path, default=RUTA_DEFECTO_NAB,
        help="Ruta al repositorio local de NAB.",
    )
    args = parser.parse_args()

    runner = NABRunner(nab_root=args.nab_path)
    fabrica_detector, nombre_detector = build_factory(
        args.detector, args.bucket_minutes, args.k_window, args.tau,
        args.cal_window
    )

    print("=" * 60)
    print(f"Iniciando evaluación de '{nombre_detector}' en NAB...")
    print("=" * 60)

    puntajes = runner.evaluar(
        detector_factory=fabrica_detector,
        detector_name=nombre_detector,
    )

    print("\n" + "=" * 60)
    print(f"Resultados oficiales de NAB para '{nombre_detector}':")
    print("=" * 60)
    for perfil, score in puntajes.items():
        print(f"  * {perfil:28s}: {score:6.2f}")
    print("=" * 60)

    # Reporte aislado: jamás mutar el final_results.json oficial de NAB.
    dest_results = ROOT_DIR / "results"
    dest_results.mkdir(parents=True, exist_ok=True)
    isolated = dest_results / f"{nombre_detector}.json"
    with open(isolated, "w", encoding="utf-8") as f:
        json.dump(
            {"detector": nombre_detector, "scores": puntajes,
             "config": {"k_window": args.k_window, "tau": args.tau,
                        "bucket_minutes": args.bucket_minutes,
                        "cal_window": args.cal_window}},
            f, indent=4, sort_keys=True,
        )
    print(f"Reporte aislado guardado en: {isolated}")
    print("NOTA: final_results.json oficial de NAB no fue modificado.")


if __name__ == "__main__":
    main()
