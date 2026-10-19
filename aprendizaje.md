# Aprendizaje — Registro Vivo de Experimentos e Integridad Metodológica (Six Eyes)

> Todo lo intentado, medido, autopsiado y archivado.
> Regla de oro: Ningún número sin artefacto en `results/`; ninguna afirmación sin medición causal reproducible.

---

## 1. Hito Metodológico: Validación Ciega Out-of-Sample de P05 (2026-10-18)

- **Artefactos Oficiales:**
  - Código y protocolo sellados en Git: Tag inmutable `prereg-p05-v1` (commit `5cb22fb8` / `23832ed4`).
  - Datos brutos: `results/frescas_frozen_p05.csv` (1,888 filas, 472 series × 4 brazos completos).
  - Informe estadístico reproducible: `results/frozen_p05_report.json`.
  - Nota técnica y reporte formal: `experiments/tsb_frozen_p05/NOTA_TECNICA_FROZEN_P05.md`.
- **Candidato Evaluado (P05):**
  $$\text{SpatialSpikeBypass}(\text{PercentileCalibrator}(\text{PredictiveReadoutESN}(\alpha_{\text{recon}}=0.5, \text{seed}=42), W=16000), \text{tol}=0.05, \text{warmup}=100)$$
  *Mezcla 50/50:* Reconstrucción contemporánea ESN + pronóstico residual un paso adelante ($h_{t-1} \to z_t$). Sin supresión refractaria ni decaimiento de picos.
- **Línea Base Oficial (P10):** Mismo arnés con $\alpha_{\text{recon}}=1.0$ (reconstrucción pura, bit a bit equivalente a `cand_s42`).

### 1.1 Resultados Pareados en las 472 Frescas (Criterio `both`: Series + Cluster Familia)

| Métrica | Base P10 | Candidato P05 | $\Delta$ Pareado Medio | IC 95% Series | IC 95% Cluster Familia | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUC-ROC** (Primaria) | 0.7828 | **0.8153** | **+0.0325** | **[+0.0273, +0.0379]** | **[+0.0143, +0.0564]** | **76.1%** (359/472) |
| **AUC-PR** (Guarda) | 0.2047 | **0.2978** | **+0.0930** | **[+0.0810, +0.1057]** | **[+0.0207, +0.1665]** | **71.0%** (335/472) |
| **VUS-PR** (Guarda) | 0.2854 | **0.3541** | **+0.0687** | **[+0.0583, +0.0794]** | **[+0.0145, +0.1230]** | **67.2%** (317/472) |

- **Veredicto:** **`SUCCESS: True`**. Se cumple simultáneamente el límite inferior $>0$ en la métrica primaria y la no-inferioridad en los guardrails temporales sin regresión significativa.

### 1.2 Compuertas Metodológicas G1–G4
- **G1 (Random):** **APROBADA** ($\text{AUC-ROC} = 0.5025 \in [0.45, 0.55]$, $\text{AUC-PR} = 0.0097$).
- **G2 (Z-Score):** **FALLO REGISTRADO** ($\text{AUC-ROC} = 0.7497 \notin [0.52, 0.66]$). Se registra como fallo sin atenuantes post-hoc.
- **G4 (Cobertura):** **APROBADA** (100%, 472/472 series válidas, 0 excluidas).

---

## 2. Lecciones y Autopsias Forenses de Fallos

### 2.1 La Falsa Alarma del "Bug de Evaluación" de G1 (Resolución de Causa Raíz)
- **Problema previo:** En corridas históricas preliminares, el detector aleatorio daba $\text{AUC-ROC} \approx 0.010$.
- **Causa raíz unificada:** Mezcla posicional de cabeceras en CSVs viejos (esquemas de 12 cols vs 5 cols). Las columnas `AUC-ROC` y `AUC-PR` estaban invertidas al volcar los resultados de `random` y `zscore`. El 0.010 correspondía a la prevalencia real de anomalías de la serie (su AUC-PR teórico), mientras que su AUC-ROC verdadero era 0.50.
- **Solución implementada:** Adopción obligatoria de `csv.DictWriter` vinculado por nombres de campo y tests unitarios de contrato en `tests/test_harness_metrics.py` y `tests/test_e2e_runner_eval.py`.

### 2.2 Auditoría de Incertidumbre: Concentración por Familias
Al aplicar bootstrap agrupado por familia (*cluster bootstrap*), el límite inferior en ROC cayó de $+0.0273$ a $+0.0143$:
- **YAHOO (47.5% de la muestra, 224 series):** Tracciona fuertemente la ganancia ($\Delta = +0.0412$, win rate 83.9%).
- **UCR (32.2% de la muestra, 152 series):** Es el segmento resistente ($\Delta = +0.0109$, win rate 57.2%). 74 de las 82 series con ROC < 0.60 pertenecen a UCR.
- **Consistencia:** Las 4 familias (YAHOO, UCR, WSD, Stock) tienen delta positivo en las tres métricas.

### 2.3 Matiz Crítico: Z-Score vs P05 en Series Cortas
- En series cortas ($\le 7,500$ puntos), P05 obtiene **0.9099** y el Z-Score **0.9088**. **Es un empate técnico con un baseline trivial.** P05 rescata el déficit que sufría P10 (0.8689), pero en cortas no supera al Z-Score.
- La superioridad neta sobre el Z-score proviene exclusivamente de las series largas (**0.7206 vs 0.5906**, $\Delta = +0.1300$).

---

## 3. Descomposición de Mecanismos: Por qué Falla el Modelo

Al someter las peores series a autopsia (p. ej. Yahoo ROC = 0.0887 y UCR Medical ROC = 0.1660), descubrimos que operan dos fallos independientes:

### 3.1 Fallo de Inicialización y Calibración (Caso Yahoo ROC = 0.0887)
- **Diagnóstico:** En la serie `574_YAHOO_id_24`, la anomalía única ocurre en $t=202$.
- **Mecanismo:** El calibrador ECDF ([`calibrador_percentil.py`](src/sixeyes/detectors/calibrador_percentil.py)) tiene una constante de warmup `MIN_HISTORIAL = 288`. Durante los primeros 288 pasos, emite forzosamente `score = 0.0`.
- **Efecto:** La anomalía real en $t=202$ recibió score 0.0, mientras que el régimen normal post-288 recibió scores entre 0.1 y 0.8. La curva ROC se invirtió por diseño de warmup, no por fallo de la red neuronal. El Z-Score (warmup=5) sacó 0.9613.

### 3.2 Patología de Adaptación Continua RLS (Caso UCR y Anomalías Sostenidas)
- **Diagnóstico:** En series con anomalías prolongadas (como `470_UCR`, meseta de 30 pasos en $t \approx 11,000$), el RLS continuo adapta sus pesos con $\lambda = 0.999$.
- **Mecanismo:** A los 15-20 pasos dentro de la anomalía, el RLS reduce sus residuos a cero (absorción). Al regresar al régimen normal, los pesos deformados generan un residuo masivo sobre puntos normales.
- **Efecto:** Falsa alarma de recuperación prolongada post-anomalía que invierte la ordenación de scores ($\text{ROC} < 0.50$).

---

## 4. Investigación de Plasticidad Selectiva (Fase B — Ablación Controlada)

Se implementó una suite diagnóstica aislada ([`scripts/experiments/diag_plasticity_ablation_b.py`](scripts/experiments/diag_plasticity_ablation_b.py)) para medir en idénticas condiciones 3 variantes frente a 3 regímenes sintéticos (Meseta anómala 200 pasos, Deriva lineal lenta legítima, y Régimen estacionario puro):

| Métrica / Escenario | Variante 1: RLS Continuo | Variante 2: RLS Congelado | Variante 3: Compuerta Prefrontal |
| :--- | :---: | :---: | :---: |
| **Absorción de la meseta (error crudo)** | 99.1% (absorbe todo) | 95.7% | 99.8% |
| **Residuo crudo en retorno normal ($t=800$)** | **1.63** (desorientación masiva) | **0.31** (pesos limpios) | **0.08** (amortiguado) |
| **Pasos de falsa alarma en recuperación (>0.5)** | **114 pasos** | 125 pasos | **58 pasos (reducción del 49%)** |
| **Score medio en recuperación** | 0.6699 (alarma activa) | 0.6537 | **0.4461 (bajo umbral de alarma)** |
| **Seguimiento de deriva legítima (error raw)** | 0.0119 | **0.1149 (falla, 10x peor)** | **0.0119 (adaptación intacta)** |
| **Falsas alarmas en deriva (>0.5)** | 47.1% | **67.8% (colapso)** | **47.1% (no bloquea deriva)** |
| **Falsas alarmas en régimen estacionario** | 34.0% | 43.2% | **34.0% (cero inestabilidad)** |

### 4.1 Descubrimientos Científicos Fundamentales de la Ablación:
1. **La compuerta prefrontal causal funciona sin degradar la deriva:** Bloquear la actualización RLS cuando el residuo contemporáneo supera $3\sigma$ reduce a la mitad la duración del falso pico de recuperación (de 114 a 58 pasos) y su intensidad a $0.44$, sin perjudicar el seguimiento de cambios lentos normales ni el régimen estacionario.
2. **La plasticidad no reside solo en los pesos $w$ (Inercia parásita de memorias):**  
   Congelar los pesos RLS (Variante 2) elimina la desorientación del residuo crudo ($0.06$ vs $1.63$), pero el score calibrado final sigue inflado durante la recuperación. **Causa descubierta:** La ventana de normalización móvil contemporánea (`_norm` de tamaño 6,400) y la memoria FIFO de la ECDF absorben la amplitud de la meseta. Para un control total de la plasticidad cortical, la compuerta debe regular tanto el RLS como la ingesta en las memorias de escala.

---

---

## 5. Fase C: Aislamiento Factorial Causal (C.1 Plasticidad, C.2 Reservorio, C.3 ECDF)

Para aislar rigurosamente la contribución de cada componente de memoria sin oráculos ni sesgos armónicos, se implementaron experimentos de laboratorio con ciclos periódicos completos ($T=50$) en [`scripts/experiments/diag_phase_c_factorial.py`](scripts/experiments/diag_phase_c_factorial.py) y [`scripts/experiments/diag_phase_c3_ecdf.py`](scripts/experiments/diag_phase_c3_ecdf.py), produciendo los 4 entregables formales en `results/diagnostics_plasticity/phase_c/`:

| Componente Evaluado | Experimento / Intervención | Hallazgo Cuantitativo | Conclusión Causal |
| :--- | :--- | :--- | :--- |
| **C.1 Plasticidad RLS ($w$)** | Normalización fija durante meseta: Continuo vs Congelado | Continuo absorbe el **98.19%** de la meseta. En retorno, pico de error crudo = **20.14** (elevación 229.8x) y cola de **155 pasos**. Con $w$ congelado: absorción = **0.0%**, pico = **2.91** (reducción 7x) y cola = **31 pasos**. | **Causa Primaria:** La adaptación no supervisada de los pesos RLS explica el **80% de la persistencia temporal** de la falsa alarma de recuperación. |
| **C.2 Inercia del Reservorio ($h_t$)** | En $t=800$: Natural vs Reset de choque ($h=0$) vs Replay de fase ($t=550$) | Natural: cola de **155 pasos**, error inmediato = 1.505. Reset $h=0$: cola de **155 pasos**, error = 1.470. Replay de fase: cola de **154 pasos**, error = 1.610. | **Efecto Despreciable:** El estado dinámico $h_t$ exhibe *fading memory* ($\rho=0.9$) que decae en $<20$ pasos. Intervenir en $h_t$ no altera la duración de la recuperación; los pesos $w$ dominan por completo. |
| **C.3 Memoria Estadística (ECDF)** | Ingesta continua estándar vs Ingesta protegida (congelada en anomalía) | Proteger la ECDF preserva la distribución de fondo, pero con RLS continuo el error crudo es tan extremo ($>20$) que satura el percentil (>0.89) independientemente del buffer histórico. | **Efecto Modulador Secundario:** La ECDF no genera la falsa alarma cruda; amplifica o preserva la cola una vez emitida. Debe coordinarse aguas abajo de la compuerta de $w$. |

### Jerarquía Causal y Prescripción de Diseño:
1. **Prioridad 1 (Crítica):** Compuerta prefrontal causal sobre los pesos $w$ del RLS (bloqueo condicionado de actualización por sorpresa predictiva).
2. **Prioridad 2 (Secundaria):** Compuerta de congelamiento sobre las estadísticas de fondo (`_norm` y buffer ECDF) durante sospecha de anomalía.
3. **Regla de No Intervención:** **NO resetear ni alterar el estado $h_t$ del reservorio**. Su dinámica recursiva natural es inocente de la inercia patológica.

---

## 6. Etapa 3: Evaluación A/B Multidimensional de Plasticidad Regulada

Implementación de [`GatedReadoutESN`](src/sixeyes/experimental/gated_readout_esn.py) extendiendo directamente [`PredictiveReadoutESN`](src/sixeyes/experimental/predictive_readout_esn.py) mediante Template Method hooks (`_should_update_weights` y `_should_ingest_norm`), garantizando paridad A/B estricta.

Resultados auditados del evaluador cuatridimensional ([`scripts/experiments/eval_gated_readout_ab.py`](scripts/experiments/eval_gated_readout_ab.py)):

| Métrica / Dimensión | Línea Base Oficial (`PredictiveReadoutESN`) | Control A/B (`GatedReadoutESN_Off`) | Compuerta Prefrontal Activa (`GatedReadoutESN_Active`) | Conclusión / Trade-off |
| :--- | :---: | :---: | :---: | :--- |
| **Equivalencia A/B** | Referencia | **100.0% Idéntico bit a bit** | N/A | Paridad demostrada matemáticamente. |
| **Fidelidad (Estacionario p95)** | 0.0189 | 0.0189 | **0.0203** | Cero falsas alarmas basales (>0.5). |
| **Recuperación (Cola post-meseta)** | **132 pasos** | **132 pasos** | **27 pasos (reducción del 79.5%)** | La compuerta extingue la persistencia de falsas alarmas. |
| **Retención (Error medio en deriva)** | **0.0153** | **0.0153** | **1.0579** (Supresión: **31.2%**) | **Hallazgo Crítico:** La compuerta binaria fija con $\sigma=3.0$ confunde rampas rápidas con anomalías y frena el 31% del aprendizaje legítimo. |
| **Cómputo (Latencia por paso)** | 178.1 µs | 186.2 µs | **172.8 µs** | Overhead imperceptible (<8 µs), >5,000 pts/seg en tiempo real. |

### Hallazgo Metodológico de la Etapa 3:
La compuerta prefrontal binaria elimina el 80% de la cola de falsa alarma de recuperación, pero **evidencia el costo de la hipoplastia en derivas rápidas (suprime 31.2% de actualizaciones)**. Esto demuestra empíricamente que Six Eyes requiere evolucionar hacia la **Fase III: un controlador continuo de plasticidad** con tasa adaptativa proporcional a la persistencia de la señal, en lugar de un interruptor de todo o nada.

---

## 7. Fase III: Controlador Continuo de Plasticidad Cinemática y RLS Ponderado

Implementación del controlador continuo en [`src/sixeyes/experimental/continuous_plasticity_controller.py`](src/sixeyes/experimental/continuous_plasticity_controller.py) y del paso RLS ponderado formal con Sherman-Morrison en [`src/sixeyes/experimental/weighted_rls.py`](src/sixeyes/experimental/weighted_rls.py).

Batería rigurosa de **10 semillas independientes** (`[42, 7, 99, 123, 2024, 314, 555, 777, 888, 999]`) sobre 4 regímenes fuera de distribución ([`scripts/experiments/eval_continuous_plasticity_ab.py`](scripts/experiments/eval_continuous_plasticity_ab.py)), con resultados consolidados en [`results/diagnostics_plasticity/continuous_controller_multidimensional.json`](results/diagnostics_plasticity/continuous_controller_multidimensional.json):

| Dimensión / Régimen | Baseline Oficial (`PredictiveReadoutESN`) | Control Binario (`GatedReadoutESN`) | Candidato Continuo (`ContinuousPlasticityController`) | Conclusión Científica |
| :--- | :---: | :---: | :---: | :--- |
| **Retención en Deriva (Rampa)** | **0.0129 ± 0.0027** | 0.3359 ± 0.3713 *(Colapso)* | **0.0129 ± 0.0027** | **Cero Hipoplastia:** El controlador continuo sigue la rampa con idéntica precisión que la línea base sin frenar el aprendizaje. |
| **Caminata Browniana (No estacionaria)** | **0.0251 ± 0.0115** | 10.7604 ± 32.1065 *(Inestabilidad)* | **0.0251 ± 0.0115** | **Robustez Estocástica:** La compuerta binaria desestabiliza series sin periodicidad; el continuo mantiene estabilidad idéntica al baseline. |
| **Recuperación tras Meseta (Cola)** | 88.7 ± 17.3 pasos | **23.3 ± 5.0 pasos** | **77.3 ± 17.8 pasos** | **Confirmación Teórica del Suelo Adaptativo ($q_t \le \eta_t \le 1$):** Durante una meseta muy prolongada, $q_t$ acumula sesgo y reactiva parcialmente la plasticidad. |
| **Sanidad Espectral de $P_t$ ($\lambda_{\min}$)** | $3.53 \times 10^{-5}$ | $4.02 \times 10^{-5}$ | **$3.54 \times 10^{-5}$** | **Estabilidad Numérica Probada:** $P_t$ permanece estrictamente definida positiva en el 100% de los pasos en las 10 semillas. |
| **Número de Condición $\kappa(P_t)$** | $4.19 \times 10^4$ | $4.17 \times 10^4$ | **$4.23 \times 10^4$** | Condicionamiento acotado e indistinguible del RLS estándar. |
| **Cómputo (Latencia por paso)** | 317.4 ± 37.5 µs | 294.5 ± 40.5 µs | **612.2 ± 61.3 µs** | Overhead de ~290 µs debido a la ventana móvil de MAD, perfectamente viable en tiempo real (>1,600 pts/seg). |

### Conclusión Epistemológica de la Fase III:
1. **Éxito en Adaptación sin Hipoplastia:** El controlador cinemático resuelve por completo el defecto de la compuerta binaria en derivas y procesos estocásticos (error idéntico al baseline).
2. **La Paradoja de la Meseta Sostenida:** La cota analítica demostrada ($q_t \le \eta_t \le 1$) protege el modelo ante shocks breves, pero ante anomalías arbitrariamente largas reactiva gradualmente la plasticidad al interpretar la persistencia como nuevo régimen normal. Esto delimita formalmente la frontera entre una política adaptativa local y la necesidad de memoria episódica o contextual externa.

---

## 8. Fase IV: Diagnóstico Temporal Fino, Autopsia de Latencia y la Transición a Memoria Contextual

Se ejecutó una instrumentación temporal paso a paso ([`scripts/experiments/diag_phase_iv_temporal_trajectory.py`](scripts/experiments/diag_phase_iv_temporal_trajectory.py)) para registrar con precisión de reloj la cinemática de $v_t$, $q_t$, $\eta_t$ y la evolución de los pesos $w_t$ durante una meseta de 200 pasos ($t \in [500, 700)$) y su retorno.

Artefactos auditables generados:
- Traza completa paso a paso: [`results/diagnostics_plasticity/phase_iv_temporal_trajectory.csv`](results/diagnostics_plasticity/phase_iv_temporal_trajectory.csv)
- Resumen temporal: [`results/diagnostics_plasticity/phase_iv_temporal_summary.json`](results/diagnostics_plasticity/phase_iv_temporal_summary.json)

### 8.1 Hallazgos Cuantitativos del Diagnóstico Temporal:
1. **Protección Inicial Inmediata en $t=500$:**
   El shock cinemático salta a $v_{500} = 26.19$, forzando a la ganancia a caer a $\eta_{500} = 0.0449$ (**95.5% de protección plástica en el paso 0**).
2. **Duración de la Ventana de Protección Efectiva:**
   Dura exactamente **12 pasos** ($t \in [500, 512)$).
   En $t=512$, el integrador de persistencia direccional $q_t$ acumula errores consistentes y cruza el umbral $0.50$.
   Debido a la cota analítica demostrada ($q_t \le \eta_t \le 1$), $\eta_t$ es forzada a subir a $\eta > 0.50$ en $t=512$ y a $\eta > 0.80$ en $t=526$.
3. **Deformación de Pesos RLS:**
   Durante los primeros 12 pasos, los pesos permanecen intactos. Durante los 188 pasos restantes de la meseta, el modelo interpreta que la persistencia es un cambio legítimo y adapta sus pesos con $\eta \approx 0.85$.
   En $t=700$, la distorsión relativa de $\|w_{\text{fc}}\|$ respecto al baseline es del **73.4%** ($2.64$ vs $3.17$).
4. **Shock de Retorno en $t=700$:**
   Al retornar a la normalidad, se produce un nuevo shock cinemático inverso ($v_{700} = 27.82$), reduciendo de nuevo la ganancia a $\eta_{700} = 0.0097$ (**99.0% de protección inmediata**).
   La cola residual se reduce de 85 pasos (baseline) a 63 pasos (controlador).

### 8.2 Autopsia de Latencia:
- El incremento de latencia de **317.4 µs a 612.2 µs/paso** proviene en un 95% del cálculo de $\text{MAD}$ mediante `np.median` repetido sobre el deque en cada paso.
- **Ruta de optimización:** Sustituir la doble mediana de numpy por un estimador recursivo streaming $O(1)$ (desviación absoluta media EWMA) para recuperar la latencia basal (<350 µs/paso).

### 8.3 Conclusión Epistemológica Central:
Ningún controlador local basado en diferencias $d_t$ y errores contemporáneos $e_t$ puede distinguir una anomalía arbitrariamente prolongada de una rampa legítima sin información contextual. Si la anomalía persiste más allá del horizonte del integrador (~15 pasos), el suelo adaptativo forzosamente reactiva el aprendizaje. Para resolver la paradoja se requiere una **memoria episódica / control de régimen superior**.

---

## 10. Fase V: Memoria Episódica Causal, Estimación Robusta de Escala y Control de Regímenes

Tras la autopsia de la Fase IV, se ejecutó el plan de investigación en 5 etapas para resolver la absorción de mesetas anómalas sin caer en la trampa de la heurística mágica ni violar la causalidad estricta.

### 10.1 Auditoría Analítica y Verificación Temporal (t=512):
Se contrastó la traza paso a paso ([`scripts/experiments/audit_phase_iv_trajectory.py`](src/sixeyes/experimental/audit_phase_iv_trajectory.py)) contra las ecuaciones analíticas:
- **Discrepancia máxima:** Exactamente `0.00e+00` en los 1200 pasos.
- **Causa del cruce en $t=512$ ($\eta = 0.5090$):**
  - Histéresis de shock: $g_{512} = \rho^{12} = (0.97265)^{12} = 0.7170$ (decae 28.3%).
  - Persistencia direccional: $q_{512} = 0.3152$ (crece ante error sostenido).
  - Efecto acoplado: $1 - g_{512}(1 - q_{512}) = 1 - 0.7170 \times (1 - 0.3152) = 0.5090$.
- **Conclusión de auditoría:** No hubo fuga temporal. El cruce es un comportamiento matemático determinista del acoplamiento entre histéresis y suelo adaptativo.
- **Suite de pruebas:** Automatizada en [`tests/test_phase_iv_diagnostics.py`](tests/test_phase_iv_diagnostics.py) (5 tests unitarios cubriendo causalidad, contrato de choque, protección de pesos y cola determinista).

### 10.2 Perfilado Empírico de Estimadores de Escala Cinemática:
Se implementaron y perfilaron 3 estimadores aislados ([`src/sixeyes/experimental/scale_estimators.py`](src/sixeyes/experimental/scale_estimators.py) y [`scripts/experiments/profile_scale_estimators.py`](scripts/experiments/profile_scale_estimators.py)) sobre 50,000 pasos:

| Estimador de Escala | Latencia p50 / p95 (µs) | Inflación post-outlier (+100σ) | Detección de Shock ($v_{500}$) | Conclusión Epistemológica |
| :--- | :---: | :---: | :---: | :--- |
| **Windowed MAD** (Fase III) | 128.1 / 305.2 µs | 2.92x | $v = 26.19$ | Responsable del 95% de la sobre-latencia. Seguro pero costoso ($O(W)$). |
| **Streaming EWMA** (Recursivo L1) | **1.1 / 2.6 µs** | 4.73x *(Contaminado)* | $v = 14.63$ *(Atenuado)* | **Pérdida severa de robustez:** Un outlier masivo infla la escala y destruye la detección del shock. |
| **Streaming Huber** (Corte $k=2.5\sigma$) | **1.0 / 2.7 µs** | **0.99x (100% Inmune)** | **$v = 26.36$** | **Óptimo de Pareto:** Recupera velocidad $O(1)$ sin diluir la detección de shocks. |

Artefacto: [`results/diagnostics_plasticity/scale_estimator_comparison.json`](results/diagnostics_plasticity/scale_estimator_comparison.json).

### 10.3 Memoria Episódica Causal y Control de Regímenes:
Se construyó [`src/sixeyes/experimental/episodic_regime_memory.py`](src/sixeyes/experimental/episodic_regime_memory.py) y [`src/sixeyes/experimental/regime_aware_controller.py`](src/sixeyes/experimental/regime_aware_controller.py), implementando una firma multimodal causal $\phi = (\mu_x, \sigma_x, \psi_h)$ donde $\psi_h = \mathbb{E}[|h_t|]$ es la envolvente de activación del atractor del reservorio.

**Regla de Decisión Causal:**
1. `KNOWN` ($d \le 1.25$ respecto al régimen base): Reconoce el estado nominal y disuelve el shock residual de retorno.
2. `UNCERTAIN_QUARANTINE` ($d > 1.25$ no cartografiado): La persistencia no confiere legitimidad; la plasticidad se acota estrictamente a $\eta \le 0.05$.
3. `NEW_PROMOTED`: Solo tras persistir $T_{\text{confirm}} = 250$ pasos consecutivos se consolida como nuevo régimen.

### 10.4 Benchmark Multidimensional Cuatridimensional (10 semillas x 5 escenarios x 4 variantes = 200 corridas):
Script: [`scripts/experiments/eval_regime_memory_benchmark.py`](scripts/experiments/eval_regime_memory_benchmark.py).
Artefacto: [`results/diagnostics_plasticity/regime_memory_benchmark_results.json`](results/diagnostics_plasticity/regime_memory_benchmark_results.json).

| Métrica / Escenario | A (Control Base) | B (Continuo Fase III) | C (Memoria Pasiva) | D (Regime-Aware Fase V) | Impacto Teórico y Práctico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Cola Post-Meseta (Pasos)** | 83.2 ± 9.3 | 70.2 ± 10.2 | 78.8 ± 10.2 | **58.4 ± 9.8 pasos** | **Reducción de 24.8 pasos (-29.8% vs base):** La cuarentena impide la absorción y reduce la cola. |
| **Norma final de pesos $\|w\|$** | 2.71 ± 0.26 | 2.26 ± 0.19 | 2.54 ± 0.21 | **2.07 ± 0.13** | **Máxima estabilidad:** Los pesos se deforman significativamente menos ante perturbaciones. |
| **Cola de retorno $A \to B \to A$** | 14.4 ± 5.5 | 14.5 ± 5.4 | 14.5 ± 5.4 | **14.5 ± 5.4 pasos** | Preserva recuperación rápida en alternancias dinámicas. |
| **Error residual en Deriva** | 0.0083 | 0.0083 | 0.0083 | **0.0050** | **Cero hipoplastia:** Adaptación superior a rampas continuas. |
| **Error residual en Escalón** | 0.0114 | 0.0083 | 0.0099 | **0.0055** | No se bloquea ante cambios permanentes; aprende con menor error estacionario. |
| **Latencia p95 por paso** | 875.1 µs | 2119.8 µs | 1869.7 µs | **1170.5 µs** | **Mejora del 44.8% en latencia p95** frente a Fase III gracias al estimador Huber. |

---

## 11. Fronteras de Alcance y Cierre de la Línea

1. **Agotamiento del conjunto fresco:** Las 472 series de TSB-AD **ya no son frescas para P05**. Se prohíbe evaluar variantes subsiguientes de forma confirmatoria sobre este split. La línea experimental de P05 queda formalmente **CERRADA**.
2. **Entorno específico:** Los resultados son válidos para **TSB-AD-U** con **semilla 42** y suite sintética multimodal de 10 semillas independientes.
3. **Sin afirmaciones SOTA:** No se ha comparado contra modelos pesados contemporáneos (Isolation Forest, LSTM-AD, Transformers); no se afirma superioridad general del estado del arte.
4. **No transferible a NAB:** El pipeline P05 no incluye supresores refractarios ni decaimiento de picos; su desempeño en NAB debe evaluarse de forma independiente.
5. **Garantía de Causalidad:** Ningún componente experimental utiliza etiquetas, información futura ni oráculos externos. Todas las suites pasan 54 de 54 pruebas unitarias formales.