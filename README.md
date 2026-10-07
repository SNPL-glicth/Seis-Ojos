
#Seis Ojos


> **Detección de anomalías en series de tiempo en modo streaming.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-clean-black.svg)]()

---

## Descripción

**Six Eyes** es una librería en Python orientada a la detección de anomalías en series temporales bajo un enfoque de **streaming en tiempo real**.

El procesamiento opera punto a punto de forma causal: el detector únicamente tiene acceso al historial previo y a la observación actual, emitiendo para cada punto un score de anomalía acotado en el intervalo $[0, 1]$ (donde mayor valor representa una mayor probabilidad de anomalía).

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

## Pruebas

Ejecuta la suite de pruebas unitarias con `pytest`:

```bash
pytest
```

---

## Arquitectura del Proyecto

```text
sixeyes/
├── pyproject.toml        # Metadatos del paquete y configuración de empaquetado
├── README.md             # Documentación principal
├── src/
│   └── sixeyes/
│       ├── __init__.py   # Punto de entrada y descripción del paquete
│       ├── detectors/    # Protocolos e implementaciones de detección
│       └── evaluation/   # Módulo de métricas y evaluación con benchmarks
└── tests/
    └── test_interfaz.py  # Pruebas de conformidad con el protocolo Detector
```

---

## Contrato del Detector

Todos los detectores deben implementar el protocolo streaming [`Detector`](src/sixeyes/detectors/base.py):

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Detector(Protocol):
    def update(self, value: float, timestamp: str) -> float:
        """Devuelve un score en [0, 1], mayor valor significa más anómalo;
        el detector solo puede usar el pasado y el punto actual."""
        ...
```


