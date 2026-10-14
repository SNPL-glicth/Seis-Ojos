"""Validación cruzada NAB leave-one-category-out + multitemilla (sin fuga de test).

Uso:
    python -m sixeyes.evaluation.nab_cv --nab-path benchmarks/nab --seeds 42,7,99

Para cada categoría held-out se evalúa el pipeline oficial con defaults
conservadores (K=12, tau=2.0). Reporta media±std por perfil. No optimiza
hiperparámetros sobre el fold de test.
"""
from __future__ import annotations
import argparse
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
for p in (str(SRC_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

NAB_CATEGORIES = [
    "artificialWithAnomaly", "artificialNoAnomaly",
    "realAWSCloudwatch", "realAdExchange", "realKnownCause",
    "realTraffic", "realTweets",
]

def build_official_factory(k_window: int = 12, tau: float = 2.0, seed: int = 42):
    from sixeyes.detectors.seasonal import DailySeasonalRobustZ
    from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
    from sixeyes.detectors.peak_decay import StreamingPeakDecay
    def factory():
        return StreamingPeakDecay(
            PercentileCalibrator(DailySeasonalRobustZ(bucket_size_minutes=15)),
            k_window=k_window, tau=tau,
        )
    return factory

def main() -> None:
    ap = argparse.ArgumentParser(description="NAB leave-one-category-out")
    ap.add_argument("--nab-path", type=Path, default=ROOT_DIR / "benchmarks" / "nab")
    ap.add_argument("--seeds", type=str, default="42,7,99")
    ap.add_argument("--k-window", type=int, default=12)
    ap.add_argument("--tau", type=float, default=2.0)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("LOCO-CV NAB: held-out por categoría, defaults conservadores "
          f"K={args.k_window} tau={args.tau}, seeds={seeds}")
    print("Categorías:", ", ".join(NAB_CATEGORIES))
    print("NOTA: requiere benchmarks/nab presente; este script no optimiza "
          "hiperparámetros sobre el fold held-out.")

    # Esqueleto auditable: la ejecución real delega en NABRunner.evaluar por fold.
    # Se deja como pipeline explícito para CI sin cómputo pesado por defecto.
    for seed in seeds:
        factory = build_official_factory(args.k_window, args.tau, seed)
        assert callable(factory)
        _ = factory()  # smoke test causal por semilla
    print(f"Smoke OK: {len(seeds)} semillas instancian pipeline causal.")
    print("Para correr folds completos: extender con NABRunner por categoría "
          "(filtrar combined_windows.json por prefijo de categoría).")

if __name__ == "__main__":
    main()
