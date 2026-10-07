"""Script principal para ejecutar la evaluación de Six Eyes en el benchmark oficial NAB."""

import argparse
import os
from pathlib import Path
import sys

# Configurar sys.path y PYTHONPATH para compatibilidad con multiprocessing en Windows
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
NAB_DIR = ROOT_DIR / "benchmarks" / "nab"

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

from detectors.aleatorio import RandomDetector
from detectors.zscore_robusto import RollingRobustZ
from evaluation.nab_runner import NABRunner

RUTA_DEFECTO_NAB = Path("benchmarks/nab")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de Six Eyes en NAB")
    parser.add_argument(
        "--detector",
        choices=["random", "zscore", "zscore_v2", "zscore_v3"],
        default="random",
        help="Detector a evaluar: 'random' (RandomDetector) o 'zscore' (RollingRobustZ)",
    )
    parser.add_argument(
        "--nab-path",
        type=Path,
        default=RUTA_DEFECTO_NAB,
        help="Ruta al repositorio local de NAB (por defecto: benchmarks/nab)",
    )

    args = parser.parse_args()

    runner = NABRunner(nab_root=args.nab_path)

    if args.detector == "random":
        nombre_detector = "sixeyes_random"
        fabrica_detector = lambda: RandomDetector(seed=42)
    elif args.detector == "zscore":
        nombre_detector = "sixeyes_zscore"
        fabrica_detector = lambda: RollingRobustZ()
    elif args.detector == "zscore_v2":
        nombre_detector = "sixeyes_zscore_v2"
        fabrica_detector = lambda: RollingRobustZ()
    elif args.detector == "zscore_v3":
        nombre_detector = "sixeyes_zscore_v3"
        fabrica_detector = lambda: RollingRobustZ()
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


if __name__ == "__main__":
    main()
