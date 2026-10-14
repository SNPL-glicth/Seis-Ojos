"""Ejecutor de evaluación y enlace con el benchmark oficial NAB."""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Callable

from dateutil import parser as date_parser

from sixeyes.detectors.base import Detector

DEFAULT_NAB_PATH = Path(__file__).resolve().parents[4] / "benchmarks" / "nab"
if not DEFAULT_NAB_PATH.is_dir():
    DEFAULT_NAB_PATH = Path("benchmarks/nab")


class NABRunner:
    """Ejecutor para evaluar detectores de Six Eyes sobre NAB.

    Procesa cada serie temporal del corpus punto a punto en estricto orden
    cronológico con una instancia independiente del detector, genera los CSVs
    con el formato oficial requerido y ejecuta el optimizador/scorer oficial de NAB.
    """

    def __init__(self, nab_root: str | Path = DEFAULT_NAB_PATH) -> None:
        self.nab_root = Path(nab_root).resolve()
        self.data_dir = self.nab_root / "data"
        self.labels_file = self.nab_root / "labels" / "combined_windows.json"
        self.results_dir = self.nab_root / "results"

        if not self.nab_root.is_dir():
            raise FileNotFoundError(f"No se encontró el directorio de NAB en: {self.nab_root}")
        if not self.labels_file.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de ventanas: {self.labels_file}")

    def _parsear_timestamp(self, ts_str: str) -> datetime:
        """Convierte una cadena de fecha a datetime para comparar contra ventanas de anomalía."""
        return date_parser.parse(ts_str)

    def _cargar_ventanas(self) -> dict[str, list[tuple[datetime, datetime]]]:
        """Carga y pre-parsea los intervalos de verdad fundamental para cada serie."""
        with open(self.labels_file, "r", encoding="utf-8") as f:
            raw_windows: dict[str, list[list[str]]] = json.load(f)

        ventanas_parseadas: dict[str, list[tuple[datetime, datetime]]] = {}
        for rel_path, windows in raw_windows.items():
            norm_rel_path = rel_path.replace("\\", "/")
            ventanas_parseadas[norm_rel_path] = [
                (self._parsear_timestamp(w[0]), self._parsear_timestamp(w[1]))
                for w in windows
            ]
        return ventanas_parseadas

    def generar_detecciones(
        self,
        detector_factory: Callable[[], Detector],
        detector_name: str,
    ) -> list[Path]:
        """Ejecuta el detector sobre todas las series temporales de NAB.

        Por cada serie se crea una instancia NUEVA del detector, garantizando
        que ningún detector retenga estado entre archivos y que las etiquetas
        se añadan a posteriori en el runner sin exposición al detector.
        """
        ventanas_por_archivo = self._cargar_ventanas()
        archivos_generados: list[Path] = []

        print(f"Generando detecciones para '{detector_name}' en {len(ventanas_por_archivo)} series...")

        for rel_path_str, ventanas in ventanas_por_archivo.items():
            csv_in = self.data_dir / rel_path_str
            if not csv_in.is_file():
                print(f"Aviso: archivo de datos no encontrado {csv_in}, omitiendo.")
                continue

            # Instancia nueva e independiente para cada archivo
            detector = detector_factory()

            # Estructura requerida por NAB: results/<detector>/<categoria>/<detector>_<nombre>.csv
            rel_path = Path(rel_path_str)
            categoria = rel_path.parent
            nombre_archivo = f"{detector_name}_{rel_path.name}"
            out_file = self.results_dir / detector_name / categoria / nombre_archivo
            out_file.parent.mkdir(parents=True, exist_ok=True)

            filas_salida: list[tuple[str, float, float, int]] = []

            with open(csv_in, "r", encoding="utf-8") as f_in:
                reader = csv.reader(f_in)
                header = next(reader)
                idx_ts = header.index("timestamp")
                idx_val = header.index("value")

                for row in reader:
                    ts_str = row[idx_ts]
                    val = float(row[idx_val])

                    # 1. El detector procesa de forma estrictamente causal
                    score = detector.update(val, ts_str)

                    # 2. El runner asigna la etiqueta para la posterior evaluación
                    ts_dt = self._parsear_timestamp(ts_str)
                    label = 1 if any(t1 <= ts_dt <= t2 for t1, t2 in ventanas) else 0

                    filas_salida.append((ts_str, val, score, label))

            # Escribir archivo en formato oficial de resultados de NAB
            with open(out_file, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.writer(f_out)
                writer.writerow(["timestamp", "value", "anomaly_score", "label"])
                writer.writerows(filas_salida)

            archivos_generados.append(out_file)

        print(f"Completadas {len(archivos_generados)} series para '{detector_name}'.")
        return archivos_generados

    def ejecutar_scorer_oficial(self, detector_name: str) -> dict[str, float]:
        """Invoca el optimizador y scorer oficial de NAB (Runner).

        Blindaje: hace backup de ``final_results.json`` y lo restaura al salir,
        porque ``Runner.normalize()`` lo reescribe en disco. Los scores se
        leen de los ``*_scores.csv`` propios (nombres con perfiles oficiales),
        no del JSON global — inmune al colapso de claves por puntos.
        """
        nab_str = str(self.nab_root)
        if nab_str not in sys.path:
            sys.path.insert(0, nab_str)

        # En pandas moderno (>= 2.0), Series.values devuelve una vista de solo lectura.
        # En CorpusLabel.getLabels se realizaba 'labels["label"].values[...] = 1',
        # lo cual genera ValueError. Aplicamos un adaptador temporal en memoria para
        # usar '.loc', evitando alterar el código fuente en disco de NAB.
        import nab.labeler as labeler  # type: ignore
        from nab.runner import Runner  # type: ignore

        orig_getLabels = labeler.CorpusLabel.getLabels

        def _compat_getLabels(self):
            import pandas

            self.labels = {}
            for relativePath, dataSet in self.corpus.dataFiles.items():
                if relativePath in self.windows:
                    windows = self.windows[relativePath]
                    labels = pandas.DataFrame({"timestamp": dataSet.data["timestamp"]})
                    labels["label"] = 0
                    for t1, t2 in windows:
                        moreThanT1 = labels[labels["timestamp"] >= t1]
                        betweenT1AndT2 = moreThanT1[moreThanT1["timestamp"] <= t2]
                        indices = betweenT1AndT2.loc[:, "label"].index
                        labels.loc[indices, "label"] = 1
                    self.labels[relativePath] = labels

        labeler.CorpusLabel.getLabels = _compat_getLabels

        final_results_file = self.results_dir / "final_results.json"
        backup: str | None = None
        if final_results_file.is_file():
            backup = final_results_file.read_text(encoding="utf-8")

        try:
            print(f"Inicializando Runner oficial de NAB para '{detector_name}'...")
            runner = Runner(
                dataDir=str(self.data_dir),
                labelPath=str(self.labels_file),
                resultsDir=str(self.results_dir),
                profilesPath=str(self.nab_root / "config" / "profiles.json"),
                thresholdPath=str(self.nab_root / "config" / "thresholds.json"),
                numCPUs=None,
            )
            runner.initialize()

            print(f"Optimizando umbrales de NAB para '{detector_name}'...")
            thresholds = runner.optimize([detector_name])

            print(f"Calculando scores oficiales de NAB para '{detector_name}'...")
            runner.score([detector_name], thresholds)

            print(f"Normalizando resultados frente a 'null'...")
            runner.normalize()
        finally:
            labeler.CorpusLabel.getLabels = orig_getLabels

        # Leer scores normalizados de nuestros CSVs propios (inmune al
        # colapso de claves de final_results.json por puntos/guiones).
        import pandas as pd

        perfiles = ["standard", "reward_low_FP_rate", "reward_low_FN_rate"]
        puntajes: dict[str, float] = {}
        det_dir = self.results_dir / detector_name
        with open(self.nab_root / "config" / "profiles.json", "r", encoding="utf-8") as f:
            profiles = json.load(f)
        with open(self.labels_file, "r", encoding="utf-8") as f:
            tp_count = sum(len(v) for v in json.load(f).values())
        null_dir = self.results_dir / "null"
        for perfil in perfiles:
            csv_path = det_dir / f"{detector_name}_{perfil}_scores.csv"
            null_csv = null_dir / f"null_{perfil}_scores.csv"
            df = pd.read_csv(csv_path)
            base = float(pd.read_csv(null_csv)["Score"].iloc[-1])
            perfect = tp_count * profiles[perfil]["CostMatrix"]["tpWeight"]
            puntajes[perfil] = float(100 * (float(df["Score"].iloc[-1]) - base) / (perfect - base))

        # Restaurar el final_results.json oficial: Runner.normalize() lo reescribe.
        if backup is not None:
            final_results_file.write_text(backup, encoding="utf-8")

        return puntajes

    def evaluar(
        self,
        detector_factory: Callable[[], Detector],
        detector_name: str,
    ) -> dict[str, float]:
        """Flujo completo: genera detecciones e invoca el scorer oficial."""
        self.generar_detecciones(detector_factory, detector_name)
        return self.ejecutar_scorer_oficial(detector_name)
