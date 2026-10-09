# Six Eyes

**Streaming Anomaly Detection for Infrastructure & Time Series Telemetry**

Six Eyes es una librería de detección de anomalías en tiempo real diseñada para flujos de datos continuos (telemetría de servidores, métricas de red y señales de sensores). Opera de forma estrictamente causal ($O(1)$ amortizado), sin fuga de datos del futuro.

---

## Modelos Principales

1. **Subespacio SVD con Buffer Gating (`SubspaceResidualDetector`):** Modela la dinámica latente mediante incrustación causal de retardos y SVD periódico. Su política de ingesta protegida ($s < 0.70$) previene que las fallas sostenidas contaminen el subespacio normal (**0.3572 VUS-PR** en TSB-AD, **0.9561 en Exathlon**, **0.7398 en SMD**).
2. **Z-Score Robusto (`RollingRobustZ`):** Estimador no paramétrico basado en mediana y MAD para detectar picos instantáneos sin sufrir sesgos por valores atípicos.
3. **Calibrador Percentil (`PercentileCalibrator`):** Mapea scores locales a una función de distribución empírica uniforme $[0, 1]$.

---

## Resultados en Benchmarks Oficiales

### ¿Por qué el puntaje en NAB supera los 30 puntos (32.74)?
En el **Numenta Anomaly Benchmark (NAB)**, donde un solo umbral binario global evalúa 58 series heterogéneas con severas penalizaciones asimétricas por falsas alarmas, **`sixeyes_zscore_pct` alcanza 32.74 (Standard)** y **40.79 (Low FN)**.
* **Mecanismo:** El calibrador por percentiles uniformiza la rareza estocástica entre series dispares. Esto permite que el optimizador de NAB fije un umbral de cola extremo ($\theta = 0.9982$) que neutraliza el 99.8% del ruido de fondo en todo el corpus, evitando penalizaciones fatales y capturando con precisión los eventos reales.

| Benchmark | Métrica Clave | Mejor Modelo | Puntaje Oficial |
| :--- | :--- | :--- | :---: |
| **TSB-AD (43 series)** | VUS-PR global | `SubspaceResidualDetector` | **0.3572** (SMD: 0.7398 / Exathlon: 0.9561) |
| **NAB (58 series)** | Standard Score | `sixeyes_zscore_pct` | **32.74** (Low FN: 40.79) |

---

## Instalación y Uso Rápido

```bash
pip install -e .
```

Ejecutar benchmarks oficiales de investigación:
```bash
python run_nab.py --detector zscore_pct      # Evaluación oficial NAB
python run_subspace_benchmark.py             # Evaluación TSB-AD (43 series)
```

Los resultados consolidados se almacenan automáticamente en `results/`.
