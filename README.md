# Six Eyes

**Streaming Anomaly Detection for Infrastructure & Time Series Telemetry**

Six Eyes es una librería experimental de detección de anomalías en tiempo real para telemetría (métricas de infraestructura, servidores, sensores). Procesamiento streaming punto a punto, sin fuga del futuro, memoria acotada.

---

## Pipeline Oficial

```
EchoStatePredictor (ESN-RLS, d=16, Nres=64, ρ=0.9, λ=0.999)
  → PercentileCalibrator (ECDF W=16000)
    → StreamingPeakDecay (K=24, τ=2.0, NMS causal)
      → SpatialSpikeBypass (tol=5%, warmup=100) — score 1.0 si rompe el rango histórico
        → RefractorySuppressor (W=80, trigger=0.990) — silencio causal post-pico
```

**Coste:** ~0.11 ms/punto (~9k pts/s monohilo). Apto en muestreos de minutos.

Componentes causales O(1) cada uno. El bypass cubre rupturas de rango; el refractario silencia réplicas y permite al umbral global descender a 0.99077.

---

## Resultados NAB (58 series)

| Medición | Valor |
| :--- | :---: |
| Histórico mejor semilla (seed 42) | 60.81 |
| **Media 3 semillas (in-sample)** | **57.68 ± 3.08** |
| **LOCO 6 folds (out-of-sample)** | **55.72 ± 10.52** |
| **Ensemble determinista + LOCO (verídico)** | **54.45 ± 13.37** |
| Base ESN sin capas | 48.30 ± 1.44 |

**Puntaje verídico: 54.45.** Es el único número determinista (sin lotería de semilla) medido fuera de muestra. El 60.81 se conserva como histórico pero es la mejor de 3 semillas in-sample. Con la media (57.68), empate con Earthgecko Skyline (58.20) y KNN-CAD (57.99) dentro del ruido ±3.08.

### Descomposición del 60.81 (seed 42)

| Pieza | Puntos |
| :--- | :---: |
| Regla de rango sola (trivial + bypass) | 50.83 |
| Cualquier predictor encima | +7.5 |
| Suerte de semilla 42 | +0.30 |
| Refractario | +2.01 (solo seed 42) |
| **Total** | **60.81** |

La regla trivial "rompió el rango" **le gana al ESN entero** (50.83 vs 49.05). NAB lo mueven las rupturas de rango, no el predictor. Dado el bypass, el ESN aporta +0.28 sobre doble escala — **cuál predictor elijas casi no importa**.

### LOCO pareado por semilla (full vs spatial-solo)

| Semilla | Full | Spatial-solo | Δ |
| :---: | :---: | :---: | :---: |
| 42 | 60.06 | 48.87 | +11.19 |
| 7 | 50.33 | 50.78 | −0.45 |
| 99 | 56.76 | 51.16 | +5.60 |
| **Media** | **55.72** | **50.27** | **+5.45 ± 3.36** |

El refractario baja la media in-sample (57.68 vs 58.50) pero **sube la transferencia** (+5.45 pareado, 2/3 positivo, p≈0.25). Su valor está en el umbral transferido, no en el score. Evidencia razonable, no demostración.

---

## Brecha Real vs ARTime

| | Ventanas |
| :--- | :---: |
| Ambas detectan | 89 |
| Solo ARTime | **17** |
| Solo spatial | 2 |
| Ninguna | 8 |

Las 17 son subidas moderadas **dentro** del rango (nyc_taxi, grok, TravelTime, machine_temperature). Score 0.92–0.99, bajo el umbral 0.99658. El bypass no puede verlas por construcción. Ni el ESN ni doble escala las cubren — **es un problema de señal, no de predictor**.

---

## Por Qué No Sube el Puntaje (medido, no opinado)

El techo no es falta de ideas — es estructural. Seis intentos archivados lo demuestran:

| Intento | Resultado | Por qué falla |
| :--- | :--- | :--- |
| LevelRarity (14/17 ventanas) | 55.50 (−2.18) | 418 FPs aislados cuestan más (~−13) que las ventanas (~+11) |
| PersistenceFilter (M=2) | 53.06 (−4.62) | Los FPs no son persistencia; el filtro deja pasar más ruido (726 FPs) |
| Refractario asimétrico | 59.32 (−1.49) | Los wakeups fragmentan el silencio; el optimizador sube el umbral a 0.99649 |
| time_features en full | 56.55 (−4.26) | Pesos fijos + entradas periódicas = ruido estructurado |
| Cura burn-in ECDF | 49.19 (+0.14) | Rescata jumpsup pero paga FPs en series limpias |
| Ensemble (59.35 in-sample) | 54.45 LOCO | El max preserva picos que no transfieren entre categorías |

**El mecanismo del techo:** NAB usa UN umbral global sobre 58 series heterogéneas. El optimizador ya encuentra el mejor tradeoff TP/FP. Cualquier señal nueva que dispare en cualquier serie suma FPs a −0.11/−0.20 cada uno. Para ganar neto, debe aportar TPs que valgan más que sus FPs — y nada de lo probado lo logra:
- Las 17 ventanas valen ~+11 pero cuestan ~+13 en FPs para capturarlas
- Bajar el umbral rescata ventanas pero admite ruido en las 58 series a la vez
- El refractario estático funciona PRECISEMENTE porque no perdona: su rigidez permite el umbral bajo 0.99077

**La única vía real:** TSB-Eva (472 series intocadas). Si el sistema generaliza fuera de NAB, es señal real; si no, el 54.45 es el techo honesto.

---


## Reproducción

```bash
pip install -e .
python run_nab.py --detector refractory --nab-path ../NAB
python -m pytest tests/ -q --tb=no   # 240 passed, 1 skipped (PYTHONPATH=src)
```

Cada corrida NAB guarda `results/sixeyes_<config>.json` aislado. `final_results.json` oficial nunca se modifica.

Registro completo de experimentos: `aprendizaje.md`. Detectores experimentales: `src/sixeyes/experimental/`.

---

## Licencia

[MIT](LICENSE)
