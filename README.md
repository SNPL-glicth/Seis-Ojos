# Six Eyes

**Streaming Anomaly Detection for Infrastructure & Time Series Telemetry**

Six Eyes es una librería de detección de anomalías en tiempo real diseñada para flujos de datos continuos (telemetría de servidores, métricas de red y señales de sensores). Opera de forma estrictamente causal ($O(1)$ amortizado), sin fuga de datos del futuro.

---

## Modelos Principales

1. **Subespacio SVD con Buffer Gating (`SubspaceResidualDetector`):** Modela la dinámica latente mediante incrustación causal de retardos y SVD periódico. Su política de ingesta protegida ($s < 0.70$) previene que las fallas sostenidas contaminen el subespacio normal (**0.3572 VUS-PR** en TSB-AD, **0.9561 en Exathlon**, **0.7398 en SMD**).
2. **Z-Score Robusto (`RollingRobustZ`):** Estimador no paramétrico basado en mediana y MAD para detectar picos instantáneos sin sufrir sesgos por valores atípicos.
3. **Calibrador Percentil (`PercentileCalibrator`):** Mapea scores locales a una función de distribución empírica uniforme $[0, 1]$.
4. **Envolvente de Crestas y Relajación Causal (`StreamingPeakDecay`):** Preserva el score a máxima intensidad en el frente de ataque del choque o en nuevas crestas locales históricas ($[t-K, t-1]$), atenuando exponencialmente mesetas y réplicas mediante un factor $\exp(-(t - t_{\text{pico}}) / \tau)$ para suprimir ráfagas redundantes de falsas alarmas fuera de ventana.

---

## Resultados en Benchmarks Oficiales

### ¿Por qué el puntaje en NAB supera los 40 puntos (40.02)?
En el **Numenta Anomaly Benchmark (NAB)**, donde un solo umbral binario global evalúa 58 series heterogéneas con severas penalizaciones asimétricas por falsas alarmas, **`sixeyes_peak_decay` alcanza 40.02 (Standard)**, **46.80 (Low FN)** y **29.13 (Low FP)**, superando baselines como Skyline (35.69) y Windowed Gaussian (39.65).
* **Mecanismo:** Desacopla la estacionalidad circadiana en canastas de 15 min (`DailySeasonalRobustZ`), mapea los residuos a percentiles uniformes $[0, 1]$ (`PercentileCalibrator`) y aplica el envolvente causal de persistencia (`StreamingPeakDecay`, $K=12, \tau=2.0$). 
* **Captura de Ventanas de Verdad Fundamental:** Al colapsar las ráfagas redundantes de falsas alarmas fuera de ventana, el optimizador de NAB desciende su umbral operativo de forma segura, incrementando la captura de anomalías reales de **64 a 69 de las 116 ventanas de verdad fundamental (59.48%)**.

| Benchmark | Métrica Clave | Mejor Modelo | Puntaje Oficial |
| :--- | :--- | :--- | :---: |
| **TSB-AD (43 series)** | VUS-PR global | `SubspaceResidualDetector` | **0.3572** (SMD: 0.7398 / Exathlon: 0.9561) |
| **NAB (58 series)** | Standard Score | `sixeyes_peak_decay` | **40.02** (Low FN: 46.80 / Low FP: 29.13) |

---

## Instalación y Uso Rápido

```bash
pip install -e .
```

Ejecutar benchmarks oficiales de investigación:
```bash
python run_nab.py                           # Evaluación oficial NAB (modelo récord automático: 40.02)
python run_subspace_benchmark.py             # Evaluación TSB-AD (43 series)
```

Los resultados consolidados se almacenan automáticamente en `results/`.
