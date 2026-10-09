"""Script principal para ejecutar la evaluación de Six Eyes en el benchmark oficial NAB."""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys

# Configurar sys.path y PYTHONPATH para compatibilidad con multiprocessing en Windows
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
NAB_DIR = ROOT_DIR / "benchmarks" / "nab"
if not NAB_DIR.is_dir():
    NAB_DIR = ROOT_DIR.parent / "benchmarks" / "nab"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(NAB_DIR) not in sys.path:
    sys.path.insert(0, str(NAB_DIR))

# Garantizar que los procesos hijos generados por multiprocessing hereden estas rutas
existing_pp = os.environ.get("PYTHONPATH", "")
extra_paths = [str(SRC_DIR), str(NAB_DIR)]
if existing_pp:
    extra_paths.append(existing_pp)
os.environ["PYTHONPATH"] = os.pathsep.join(extra_paths)

try:
    from detectors.aleatorio import RandomDetector
    from detectors.zscore_robusto import RollingRobustZ
    from detectors.seasonal import DailySeasonalRobustZ
    from detectors.biseasonal import BiSeasonalRobustZ
    from detectors.calibrador_percentil import PercentileCalibrator
    from detectors.subespacio import SubspaceResidualDetector
    from detectors.peak_decay import StreamingPeakDecay
except ImportError:
    from src.detectors.aleatorio import RandomDetector
    from src.detectors.zscore_robusto import RollingRobustZ
    from src.detectors.seasonal import DailySeasonalRobustZ
    from src.detectors.biseasonal import BiSeasonalRobustZ
    from src.detectors.calibrador_percentil import PercentileCalibrator
    from src.detectors.subespacio import SubspaceResidualDetector
    from src.detectors.peak_decay import StreamingPeakDecay

try:
    from evaluation.nab_runner import NABRunner
except ImportError:
    from src.evaluation.nab_runner import NABRunner

RUTA_DEFECTO_NAB = NAB_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de Six Eyes en NAB")
    parser.add_argument(
        "--detector",
        choices=["peak_decay", "seasonal_pct", "biseasonal_pct", "biseasonal", "seasonal", "zscore_pct", "subspace", "zscore", "random", "zscore_v2", "zscore_v3"],
        default="peak_decay",
        help="Detector a evaluar (por defecto: 'peak_decay' = DailySeasonalRobustZ + PercentileCalibrator + StreamingPeakDecay, récord de 40.02)",
    )
    parser.add_argument(
        "--nab-path",
        type=Path,
        default=RUTA_DEFECTO_NAB,
        help="Ruta al repositorio local de NAB (por defecto: benchmarks/nab)",
    )

    args = parser.parse_args()

    runner = NABRunner(nab_root=args.nab_path)

    if args.detector == "peak_decay":
        nombre_detector = "sixeyes_peak_decay"
        fabrica_detector = lambda: StreamingPeakDecay(PercentileCalibrator(DailySeasonalRobustZ()), k_window=12, tau=2.0)
    elif args.detector == "random":
        nombre_detector = "sixeyes_random"
        fabrica_detector = lambda: RandomDetector(seed=42)
    elif args.detector == "zscore":
        nombre_detector = "sixeyes_zscore"
        fabrica_detector = lambda: RollingRobustZ()
    elif args.detector == "zscore_v2":
        nombre_detector = "sixeyes_zscore_v2"
        fabrica_detector = lambda: RollingRobustZ()
    elif args.detector == "zscore_pct":
        nombre_detector = "sixeyes_zscore_pct"
        fabrica_detector = lambda: PercentileCalibrator(RollingRobustZ())
    elif args.detector == "zscore_v3":
        nombre_detector = "sixeyes_zscore_v3"
        fabrica_detector = lambda: RollingRobustZ()
    elif args.detector == "biseasonal_pct":
        nombre_detector = "sixeyes_biseasonal_pct"
        fabrica_detector = lambda: PercentileCalibrator(BiSeasonalRobustZ())
    elif args.detector == "biseasonal":
        nombre_detector = "sixeyes_biseasonal"
        fabrica_detector = lambda: BiSeasonalRobustZ()
    elif args.detector == "seasonal_pct":
        nombre_detector = "sixeyes_seasonal_pct"
        fabrica_detector = lambda: PercentileCalibrator(DailySeasonalRobustZ())
    elif args.detector == "seasonal":
        nombre_detector = "sixeyes_seasonal"
        fabrica_detector = lambda: DailySeasonalRobustZ()
    elif args.detector == "subspace":
        nombre_detector = "sixeyes_subspace"
        fabrica_detector = lambda: SubspaceResidualDetector()
    else:
        raise ValueError(f"Detector desconocido: {args.detector}")

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

    orig_final = runner.results_dir / "final_results.json"
    if orig_final.is_file():
        dest_results = ROOT_DIR / "results"
        dest_results.mkdir(parents=True, exist_ok=True)

        # 1. Preservar copia cruda sin mutar generada por NAB
        raw_target = dest_results / "nab_raw_results.json"
        shutil.copy2(orig_final, raw_target)
        print(f"Resultado crudo original de NAB preservado en: {raw_target}")

        # 2. Generar consolidado limpio con manejo explícito de errores
        try:
            with open(orig_final, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "sixeyes" in data:
                data["sixeyes"] = {
                    "reward_low_FN_rate": puntajes.get("reward_low_FN_rate", 0.0),
                    "reward_low_FP_rate": puntajes.get("reward_low_FP_rate", 0.0),
                    "standard": puntajes.get("standard", 0.0),
                }

            consolidado_path = dest_results / "final_results.json"
            with open(consolidado_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, sort_keys=True)

            shutil.copy2(consolidado_path, ROOT_DIR / "final_results.json")
            print(f"Resultados consolidados guardados en: {consolidado_path}")

        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Error al procesar y normalizar final_results.json: {e}")
            raise


if __name__ == "__main__":
    main()
