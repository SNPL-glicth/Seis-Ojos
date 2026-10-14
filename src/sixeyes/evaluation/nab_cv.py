"""LOCO real por categoría NAB con transferencia de umbral (sin fuga de test).

Método: las detecciones se generan UNA vez (pipeline oficial, sin tunear);
para cada categoría held-out C se barre el umbral SOLO con archivos fuera
de C (maquinaria oficial `Sweeper`) y se puntúan los archivos de C con ese
umbral transferido. Normalización por fold con la misma fórmula de
`Runner.normalize` (baseline null del fold + perfect del fold).

Uso:
    python src/sixeyes/evaluation/nab_cv.py --nab-path ../NAB \
        --detector-dir sixeyes_esn_k24_tau2p0_w16000

Escribe `results/loco_<detector>.json`. No muta `final_results.json`
(lectura de CSVs propios + null; sin `Runner`).
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd
from dateutil import parser as dp

FILE = Path(__file__).resolve()
SRC_DIR = FILE.parents[2]
ROOT_DIR = FILE.parents[3]
for p in (str(SRC_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

NAB_CATEGORIES = [
    "artificialWithAnomaly", "artificialNoAnomaly",
    "realAWSCloudwatch", "realAdExchange", "realKnownCause",
    "realTraffic", "realTweets",
]


def build_official_factory(k_window: int = 24, tau: float = 2.0, seed: int = 42):
    """Fábrica del pipeline oficial vigente (ESN + calibrador + PeakDecay)."""
    from sixeyes.detectors.echo_state import EchoStatePredictor
    from sixeyes.detectors.calibrador_percentil import PercentileCalibrator
    from sixeyes.detectors.peak_decay import StreamingPeakDecay

    def factory():
        return StreamingPeakDecay(
            PercentileCalibrator(EchoStatePredictor(seed=seed)),
            k_window=k_window, tau=tau,
        )
    return factory


def load_corpus(nab_root: Path, detector_dir: str):
    """Lee detecciones propias + ventanas. Devuelve dict rel -> frame."""
    det_dir = nab_root / "results" / detector_dir
    labels = json.load(open(nab_root / "labels" / "combined_windows.json", encoding="utf-8"))
    corpus = {}
    for rel in labels:
        rp = Path(rel)
        csvp = det_dir / rp.parent / f"{detector_dir}_{rp.name}"
        if not csvp.is_file():
            raise FileNotFoundError(f"Falta CSV de detecciones: {csvp}")
        df = pd.read_csv(csvp, usecols=["timestamp", "anomaly_score"])
        corpus[rel] = {
            "timestamps": tuple(pd.to_datetime(df["timestamp"]).tolist()),
            "scores": tuple(float(x) for x in df["anomaly_score"].tolist()),
            "windows": tuple((dp.parse(a), dp.parse(b)) for a, b in labels[rel]),
        }
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser(description="NAB leave-one-category-out real")
    ap.add_argument("--nab-path", type=Path, required=True)
    ap.add_argument("--detector-dir", type=str, required=True,
                    help="Carpeta en <nab>/results con los CSVs de detección.")
    ap.add_argument("--k-window", type=int, default=24)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    nab_root = args.nab_path.resolve()
    sys.path.insert(0, str(nab_root))
    from nab.sweeper import Sweeper  # type: ignore

    # Smoke: la fábrica oficial instancia con la semilla pedida.
    assert callable(build_official_factory(args.k_window, args.tau, args.seed))

    profiles = json.load(open(nab_root / "config" / "profiles.json", encoding="utf-8"))
    corpus = load_corpus(nab_root, args.detector_dir)
    by_cat: dict[str, list[str]] = {c: [] for c in NAB_CATEGORIES}
    for rel in corpus:
        by_cat[rel.split("/")[0]].append(rel)

    # Baseline null por fold: CSVs oficiales del detector nulo.
    null_dir = nab_root / "results" / "null"
    if not null_dir.is_dir():
        raise FileNotFoundError("Falta results/null para baselines.")

    report: dict = {"detector": args.detector_dir, "folds": {}}
    for profile, spec in profiles.items():
        cm = spec["CostMatrix"]
        sweeper = Sweeper(probationPercent=0.15, costMatrix=cm)
        fold_scores = []
        for held in NAB_CATEGORIES:
            train_rels = [r for c, rs in by_cat.items() for r in rs if c != held]
            test_rels = by_cat[held]
            train_list = []
            for r in train_rels:
                d = corpus[r]
                train_list.extend(sweeper.calcSweepScore(
                    list(d["timestamps"]), list(d["scores"]), list(d["windows"]), r))
            sweep = sweeper.calcScoreByThreshold(train_list)
            best = max(sweep, key=lambda t: t.score)
            raw_test, raw_null, tp_fold = 0.0, 0.0, 0
            for r in test_rels:
                d = corpus[r]
                _, mrow = sweeper.scoreDataSet(
                    d["timestamps"], d["scores"], d["windows"], r, best.threshold)
                raw_test += mrow.score if mrow is not None else 0.0
                tp_fold += len(d["windows"])
                rp = Path(r)
                ncsv = null_dir / rp.parent / f"null_{rp.name}"
                ndf = pd.read_csv(ncsv, usecols=["timestamp", "anomaly_score"])
                _, nrow = sweeper.scoreDataSet(
                    tuple(pd.to_datetime(ndf["timestamp"]).tolist()),
                    tuple(float(x) for x in ndf["anomaly_score"].tolist()),
                    list(d["windows"]), r, best.threshold)
                raw_null += nrow.score if nrow is not None else 0.0
            perfect = tp_fold * cm["tpWeight"]
            if tp_fold == 0:
                # Fold sin ventanas (ej. artificialNoAnomaly): la
                # normalización es indefinida (perfect=0). Se registra
                # como nulo y se EXCLUYE de media/std.
                fold_scores.append(None)
                report["folds"].setdefault(held, {})[profile] = {
                    "score": None, "threshold_transferido": best.threshold,
                    "archivos": len(test_rels), "excluido_media": "sin ventanas",
                }
                continue
            norm = 100.0 * (raw_test - raw_null) / (perfect - raw_null) if perfect != raw_null else 0.0
            fold_scores.append(norm)
            report["folds"].setdefault(held, {})[profile] = {
                "score": norm, "threshold_transferido": best.threshold,
                "archivos": len(test_rels),
            }
        valid = [x for x in fold_scores if x is not None]
        report[profile] = {
            "media": statistics.mean(valid),
            "std": statistics.pstdev(valid) if len(valid) > 1 else 0.0,
            "folds": fold_scores,
            "nota": "folds nulos (sin ventanas) excluidos de media/std",
        }
        print(f"{profile:22s} media={report[profile]['media']:.2f} "
              f"std={report[profile]['std']:.2f} "
              f"folds={['nulo' if x is None else round(x, 1) for x in fold_scores]}")

    dest = ROOT_DIR / "results" / f"loco_{args.detector_dir}.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"reporte: {dest}")


if __name__ == "__main__":
    main()
