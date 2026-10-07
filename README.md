# Six Eyes

> **Detección de anomalías en series de tiempo en modo streaming.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-clean-black.svg)]()

---

## Descripción

**Six Eyes** es una librería en Python orientada a la detección de anomalías en series temporales bajo un enfoque de **streaming en tiempo real**.

El procesamiento opera punto a punto de forma estricta y causal: el detector únicamente tiene acceso al historial previo y a la observación actual. No se permite la inspección del futuro ni el recalculo retroactivo. Para cada observación temporal, el detector emite un score de anomalía acotado en el intervalo `[0, 1]`, donde un mayor valor representa una mayor probabilidad de que el punto actual sea anómalo.

---

## Instalación

Instala el paquete en modo editable dentro de tu entorno virtual:

```bash
pip install -e .
```

O incluyendo las herramientas de desarrollo y pruebas:

```bash
pip install -e ".[dev]"
```

---

## Detectores Implementados

La librería incluye distintos algoritmos de detección:

*   **RandomDetector (`aleatorio.py`)**: Genera scores aleatorios uniformemente distribuidos. Sirve como línea base (baseline) para validar que el pipeline de evaluación y alineación de marcas de tiempo funciona correctamente, y para proporcionar un piso de desempeño.
*   **RollingRobustZ (`zscore_robusto.py`)**: Calcula el z-score robusto sobre una ventana móvil. Extrae métricas de tendencia central y dispersión resistentes a valores atípicos (Mediana y Desviación Absoluta de la Mediana, o MAD). Cuenta con una estrategia de respaldo progresiva para periodos de nula variabilidad (MAD = 0) utilizando resolución observada, y un mapeo asintótico de puntuaciones para evitar la saturación exponencial.
*   **DailySeasonalRobustZ (`seasonal.py`)**: Extensión del modelo de z-score orientada a la estacionalidad diaria. Divide el día en canastas de tiempo (por defecto de 15 minutos) y mantiene un historial y contexto estadístico completamente independiente para cada canasta. Esto le permite mitigar falsos positivos causados por el comportamiento normal asociado al ciclo diario.

---

## Contrato del Detector

Todos los detectores deben implementar el protocolo streaming `Detector` expuesto en `src/detectors/base.py`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Detector(Protocol):
    """Protocolo base para detectores de anomalías en streaming."""

    def update(self, value: float, timestamp: str) -> float:
        """Devuelve un score en [0, 1], mayor valor significa más anómalo;
        el detector solo puede usar el pasado y el punto actual."""
        ...
```

---

## Uso Básico

Instanciar y consumir un detector en un flujo de datos requiere únicamente la llamada al método `update`:

```python
from detectors import RollingRobustZ

# Instanciar el detector con una memoria de 50 observaciones pasadas
detector = RollingRobustZ(window_size=50, min_observations=5)

datos_streaming = [
    (10.0, "2026-10-07 08:00:00"),
    (10.5, "2026-10-07 08:05:00"),
    (9.8,  "2026-10-07 08:10:00"),
    (150.0, "2026-10-07 08:15:00") # Evento atípico
]

for valor, timestamp in datos_streaming:
    score = detector.update(valor, timestamp)
    print(f"[{timestamp}] Valor: {valor:6.1f} | Score: {score:.4f}")
```

---

## Arquitectura del Proyecto

```text
sixeyes/
├── pyproject.toml        # Metadatos del paquete y configuración de empaquetado
├── run_nab.py            # Orquestador para evaluación en Numenta Anomaly Benchmark
├── README.md             # Documentación principal
├── src/
│   └── detectors/        # Módulo de algoritmos e interfaces
│       ├── __init__.py
│       ├── base.py
│       ├── aleatorio.py
│       ├── seasonal.py
│       └── zscore_robusto.py
├── diagnostics/          # Herramientas de diagnóstico de métricas y falsos positivos
└── tests/                # Suite de pruebas unitarias
```

---

## Evaluación y Benchmarks

Six Eyes incluye herramientas de evaluación frente a repositorios estándar como el **Numenta Anomaly Benchmark (NAB)**. Se puede iniciar una ejecución completa proporcionando el detector a usar:

```bash
python run_nab.py --detector seasonal --nab-path "../benchmarks/nab"
```

El script se encarga de crear las estructuras correspondientes, delegar la ejecución punto a punto y consolidar los scores finales a través de la infraestructura oficial de NAB.

---

## Pruebas

El código incluye una extensa batería de pruebas unitarias para certificar el protocolo causal y comportamiento determinista. Ejecuta la suite con:

```bash
pytest
```
