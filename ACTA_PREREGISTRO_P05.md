# Acta de Pre-Registro Formal: Evaluación Congelada de P05 en 472 Frescas (TSB-AD)

- **Fecha y Hora de Pre-Registro:** 2026-10-18 (antes de generar el CSV limpio de evaluación de frescas)
- **Tag de Git:** `prereg-p05-v1`
- **Script de Evaluación Inmutable:** [`evaluate_frozen_p05.py`](evaluate_frozen_p05.py)
- **Runner Canónico:** [`frozen_runner.py`](frozen_runner.py)
- **Test de Contrato y Arnés:** [`tests/test_e2e_runner_eval.py`](tests/test_e2e_runner_eval.py) y [`tests/test_harness_metrics.py`](tests/test_harness_metrics.py)

---

## 1. Definición Explícita de Brazos y Candidato Congelado

- **Candidato Congelado (`P05`):**
  $$\text{SpatialSpikeBypass}(\text{PercentileCalibrator}(\text{PredictiveReadoutESN}(\alpha_{\text{recon}}=0.5, \text{seed}=42), W=16000), \text{tol}=0.05, \text{warmup}=100)$$
  - Mezcla 50/50: error de reconstrucción contemporánea RLS ($\alpha=0.5$) y error de pronóstico residual paso adelante RLS ($1-\alpha=0.5$).
  - **Aclaración crítica:** No incluye ni `StreamingPeakDecay` ni `RefractorySuppressor`.
- **Línea Base Oficial (`P10`):**
  $$\text{SpatialSpikeBypass}(\text{PercentileCalibrator}(\text{PredictiveReadoutESN}(\alpha_{\text{recon}}=1.0, \text{seed}=42), W=16000), \text{tol}=0.05, \text{warmup}=100)$$
  - Equivale bit a bit a la base oficial `cand_s42` (reconstrucción contemporánea pura).
- **Detector Aleatorio (`random`):** `RandomDetector(seed=42)` uniforme en $[0, 1]$.
- **Z-Score Robusto (`zscore`):** `RollingRobustZ(window_size=50)`.

---

## 2. Parámetros de Pre-Registro (`PREREG`)

```python
PREREG = {
    "candidate": "P05",
    "baseline": "P10",
    "random": "random",
    "zscore": "zscore",
    "primary_metric": "auc_roc",
    "guardrail_metrics": ["auc_pr", "vus_pr"],
    "ci": 0.95,
    "n_boot": 10000,
    "boot_seed": 12345,
    "g1_range": (0.45, 0.55),
    "g2_range": (0.52, 0.66),
    "g4_min_coverage": 0.80,
    "n_expected_series": 472,
    "noninferiority_margin": 0.01,
    "decisive_ci": "both",
}
```

---

## 3. Criterio de Éxito y Regla de Veredicto

El veredicto formal declara `SUCCESS: True` únicamente si se cumplen simultáneamente las siguientes condiciones:
1. **Compuerta G4 (Cobertura):** $\ge 80\%$ de las 472 series deben ser válidas ($N_{\text{valid}} \ge 378$).
2. **Política de Exclusión Estricta:** Si una serie falla, arroja timeout o genera `NaN`/`Inf` en cualquiera de sus 4 brazos o en cualquiera de sus métricas (`auc_roc`, `auc_pr`, `vus_pr`), **queda completamente excluida** de todas las comparaciones (G1, G2, pareadas y G4).
3. **Métrica Primaria (`auc_roc`):** Límite inferior del IC 95% $> 0$ tanto en bootstrap estándar de series (`ci[0] > 0`) como en bootstrap por clusters de familia (`ci_cluster_family[0] > 0`).
4. **Métricas de Guarda (`auc_pr`, `vus_pr`):** En ambas versiones del IC (`both`):
   - Sin regresión significativa: `ci_upper >= 0`.
   - No-inferioridad respetada: `ci_lower > -0.01` (margen de caída no superior al 1%).
5. **G1 y G2:** Se reportan formalmente. Si el z-score queda fuera de $[0.52, 0.66]$, G2 se registra como **FALLO** sin atenuantes.

---

## 4. Declaración de Exposición Previa e Integridad de Datos

- **Estado previo de las 472 frescas:**
  - El candidato `P05` **nunca ha sido evaluado** sobre las 472 series frescas. La evaluación es completamente out-of-sample.
  - La base `P10` (`cand_s42`) y `zscore` fueron observados en ejecuciones históricas parciales con calidad y esquemas mezclados.
  - El archivo histórico `sixeyes_fresh257_cobertura.csv` contiene exactamente **277 series únicas evaluadas** (no 257; 257 fue la lista original de pendientes tras una primera tanda, y al completarse cubrió 277).
  - Todas las métricas previas citadas (0.783, 0.750, y los cortes corto/largo de 0.869 vs 0.909) se declaran formalmente como **provisionales e invalidadas**, y quedan a la espera del cómputo limpio bajo este protocolo.

---

## 5. Protocolo de Ejecución del Runner

1. Las 472 series se confirmaron como 100% univariadas.
2. Cada serie se evalúa una única vez computando `sliding_window` de forma idéntica para los 4 brazos.
3. El recorte en $[0, 1]$ es idéntico para todos los brazos.
4. La persistencia se realiza con `csv.DictWriter` garantizando cabeceras fijas por nombre:
   `series,family,n_points,arm,auc_roc,auc_pr,vus_pr`.
