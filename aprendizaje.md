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

---

## 12. Autopsia Experimental y Veredicto Negativo: Evaluación de Fase V en Benchmarks Reales (NAB y TSB-AD) (2026-10-18)

> **Criterio Epistemológico:** Ninguna mejora de laboratorio sintético se considera un avance real del sistema hasta ser validada en los evaluadores oficiales de NAB y TSB-AD. Si no supera al baseline congelado, se declara fallo formalmente y se documenta sin atenuantes.

### 12.1 Evaluación Oficial en NAB (58 Series, 3 Perfiles Oficiales)
Se evaluaron bajo idénticas condiciones el baseline titular oficial frente a la Fase III y la Fase V:
- **Artefactos:** `results/sixeyes_cont_plasticity_s42_k24_tau2p0_w16000.json` y `results/sixeyes_regime_aware_s42_k24_tau2p0_w16000.json`.

| Perfil Oficial NAB | Baseline Titular (`esn_k24_tau2p0`) | Fase III (`cont_plasticity`) | Fase V (`regime_aware` Huber+Memoria) | $\Delta$ Fase V vs Base | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard** | **49.05** | 45.77 | 47.43 | **-1.62** | **NO SUPERA** |
| **Reward Low FP** | 38.99 | 37.81 | **39.30** | **+0.31** | Mejora marginal en FP |
| **Reward Low FN** | **57.01** | 54.16 | 54.86 | **-2.15** | **NO SUPERA** |

**Autopsia NAB:** La Fase V recupera parcialmente el déficit de la Fase III (+1.66 sobre 45.77) gracias a la cuarentena de plasticidad ($\eta \le 0.05$), e incluso bate ligeramente al baseline en `Reward Low FP` (39.30 vs 38.99). Sin embargo, en el perfil `Standard` global (49.05 vs 47.43) queda por detrás de la línea base estática. En NAB, no representó un avance neto general.

---

### 12.2 Evaluación Oficial en TSB-AD Tuning (43 Series, Pareada vs. Campeón P05)
Para aislar rigurosamente el impacto de la Fase V sin el sesgo artificial del decaimiento de NAB, se evaluó montada directamente sobre el chasis canónico del campeón histórico **P05** (`SpatialSpikeBypass(PercentileCalibrator(alpha=0.5))`, sin `StreamingPeakDecay`):
- **Artefactos:** `results/diagnostics_plasticity/p05_vs_fase_v_t43.csv` y `results/diagnostics_plasticity/p05_vs_fase_v_t43.json`.
- **Protocolo:** Comparación pareada serie por serie contra `P05` con Bootstrap al 95% (4.000 iteraciones).

| Brazo Evaluado | Arquitectura Exacta | AUC-ROC | VUS-PR | AUC-PR | Series OK |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **P10** | ESN puro ($\alpha=1.0$, reconstrucción pura) | 0.7217 | 0.2908 | 0.2267 | 43 / 43 |
| **P05 (Campeón Histórico)** | ESN híbrido ($\alpha=0.5$, pronóstico 1-paso) | **0.7612** | **0.3389** | 0.3154 | 43 / 43 |
| **P05_FaseIII** | P05 + Plasticidad Continua ($q_t \le \eta_t \le 1$) | 0.7583 | 0.3423 | **0.3205** | 42 / 43 *(Timeout)* |
| **P05_FaseV** | P05 + Huber $k=2.5$ + Memoria $K=5$ | **0.7647** | 0.3349 | 0.3173 | **43 / 43 (100% OK)** |

#### Métricas Pareadas Pre-Registradas ($\Delta = \text{Fase V} - \text{P05}$):
* **$\Delta$ AUC-ROC:** $+0.0036$ (de $0.7612$ a $0.7647$).
* **IC 95% Bootstrap pareado:** **$[-0.0092, +0.0217]$** (el intervalo cruza el cero).
* **Win Rate pareado en AUC-ROC:** **$51.2\%$** (22 victorias, 21 derrotas).
* **$\Delta$ VUS-PR:** $-0.0040$ (IC 95%: $[-0.0156, +0.0098]$).
* **$\Delta$ AUC-PR:** $+0.0019$ (IC 95%: $[-0.0073, +0.0125]$).
* **Veredicto Formal:** **`supera_p05: False`**.

---

### 12.3 Veredicto y Cierre de la Línea Experimental de Plasticidad y Memoria
1. **Empate Técnico en TSB-AD:** El chasis de pronóstico híbrido P05 ($\alpha=0.5$) ya captura la gran mayoría de la señal de anomalía por sí solo. Añadir la capa de memoria episódica $K=5$, estimador Huber y cuarentena de plasticidad $\eta_t \le 0.05$ no produce una ganancia estadísticamente distinguible del azar (win rate 51.2%, IC cruza el cero).
2. **Ausencia de Avance Global:** No se logró batir a la línea base en el puntaje Standard de NAB ni superar de forma comprobable a P05 en TSB-AD. La complejidad algorítmica adicional no se justifica para inferencia en producción.
3. **Decisión Metodológica Ejecutada:**
   - Se revierten de forma inmediata todas las extensiones experimentales de `run_nab.py` y de los ejecutores de TSB.
   - Los scripts y runners quedan en su estado limpio original.
   - **El estado del arte interno del proyecto se mantiene congelado en P05** para TSB-AD (`0.8153` en 472 series) y en el pipeline oficial titular para NAB.
   - La línea de investigación de modulación de plasticidad RLS y memoria episódica queda formalmente **CERRADA y ARCHIVADA**.

---

### 12.4 Autopsia Exprés: Transferencia Directa de P05 a NAB Titular (FALLO CATASTRÓFICO: −9.12 puntos)

- **Fecha:** 2026-10-18
- **Hipótesis puesta a prueba:** ¿Mejora NAB si sustituimos el `EchoStatePredictor` (reconstrucción pura) del titular por el `PredictiveReadoutESN` ($\alpha=0.5$, campeón de TSB-AD) manteniendo intacto el pipeline titular (`SpatialSpikeBypass` + `RefractorySuppressor`)?
- **Tiempo de ejecución:** 151.8 s (58 series oficiales).
- **Artefacto:** [`results/sixeyes_p05_titular_s42.json`](results/sixeyes_p05_titular_s42.json).

| Perfil Oficial NAB | Titular Oficial NAB (s42) | Candidato P05-Titular (s42) | $\Delta$ Directo | Veredicto |
| :--- | :---: | :---: | :---: | :---: |
| **Standard** | **60.81** | **51.69** | **−9.12** | **FALLO SEVERO** |
| **Reward Low FP** | **43.40** | **36.38** | **−7.02** | **FALLO** |
| **Reward Low FN** | **68.45** | **62.24** | **−6.21** | **FALLO** |

#### Causa Raíz de la Falsación:
1. **Sensibilidad Excesiva a Volatilidad Normal:** La cabeza de pronóstico residual un paso adelante ($h_{t-1} \to z_t$) que eleva la discriminación en TSB-AD, en NAB sobre-reacciona a los micro-cambios de volatilidad normales en series con ruido denso (`realTweets`, `realAWS`).
2. **Inundación de Falsas Alarmas que Superan al Refractario:** El incremento en la frecuencia de picos rompe el periodo de descanso del refractario ($W_{\text{rest}}=80$). Al acumular decenas de falsas alarmas adicionales fuera de ventana, el optimizador de umbrales de NAB se ve forzado a elevar drásticamente el umbral global, provocando la pérdida neta de ventanas completas.
3. **Regla Inmutable del Proyecto:** **PROHIBIDO REPETIR ESTA TRANSFERENCIA.** P05 pertenece estrictamente a TSB-AD (sin refractario ni decaimiento). El pipeline oficial titular de NAB debe mantener su predictor de reconstrucción contemporánea pura.

---

### 12.5 Auditoría Crítica de Hipótesis: Verificación de Esquemas de Entrada y No-Linealidad de Penalizaciones (2026-10-18)

> **Criterio Epistemológico:** Prohibido proponer intervenciones algorítmicas sin verificar primero el esquema real de los datos en disco y la matemática exacta de la función de coste. Se corrigen dos diagnósticos erróneos previos:

1. **Falsación de la Hipótesis Multivariada en TSB-AD-U:**
   - **Error previo:** Se conjeturó que el rendimiento en la familia UCR caía porque `val = float(data[i, 0])` descartaba canales en series multivariadas.
   - **Verificación empírica en disco:** La colección es `TSB-AD-U` (**U**nivariate). La inspección directa de los archivos de UCR (`305_UCR_id_3_Medical`, etc.) confirmó que el esquema es estrictamente de 2 columnas: `['Data', 'Label']` (1 señal escalar + 1 etiqueta). No existe pérdida de canales ni información descartada.
   - **Regla fijada:** Se descarta de forma terminante cualquier intento de implementar PCA multivariado o normas $L_2$ de reducción. La dificultad en UCR proviene de la dinámica biológica no estacionaria y el horizonte de memoria del reservorio, no de un defecto de lectura del CSV.

2. **No-Linealidad de la Penalización por Falsas Alarmas en NAB:**
   - **Corrección matemática:** La penalización por falsos positivos en NAB no es plana a $-0.11$ constante por punto. Está ponderada por la distancia temporal a la ventana anómala precedente ($F_{\text{eff}} = \sum_{j \in \text{FP}} \sigma(d_j)$).
   - **Proyecciones lineales invalidadas:** El cálculo $382 \times -0.11 = -42.02$ asume el peor caso teórico para cada punto. Proyectar que eliminar 200 FPs sube exactamente $+9.48$ puntos ignora que el *Sweeper* re-optimiza el umbral escalar global $\theta^*$, lo que puede acarrear la pérdida de ventanas legítimas.
   - **Errata formal corregida:** Six Eyes detecta **89 de 116 ventanas** (no "muchas más" que ContextOSE, el cual detecta **96**).
   - **Rechazo a filtros heurísticos:** Se rechaza imponer umbrales ciegos de corte (p. ej. $4\sigma$) sin demostración empírica pareada sobre scores ya emitidos, ya que mutilarían anomalías sutiles dentro del rango histórico.

3. **Veredicto Operativo Final:**
   - **Cero nuevas fases experimentales:** No se abre ninguna línea de código nueva.
   - **El sistema se mantiene estrictamente congelado** en sus dos referencias auditadas:
     - **TSB-AD:** `P05` híbrido $\alpha=0.5$ (**0.8153** en 472 series).
     - **NAB:** Titular oficial `sixeyes_refractory_w80_t990` (**57.68 ± 3.08** multisemilla / **55.72** LOCO out-of-sample).

---

## 13. Autopsia Forense de la Inversión de Ranking en UCR (TSB-AD): Hipovariabilidad, Punto Ciego del Error Residual y Formulación de Dinámica Local (2026-10-19)

> **Criterio Epistemológico:** Ningún cambio monótono de escala, filtro arbitrario ni umbral heurístico puede corregir una ordenación donde los puntos anómalos reciben sistemáticamente puntuaciones inferiores a los puntos normales. Se aislaron los mecanismos numéricos exactos mediante inspección punto a punto en series con ranking colapsado.

### 13.1 Bifurcación de Casos en UCR: Ruptura de Secuencia vs. Pérdida de Dinámica
El análisis comparativo pareado entre el error de reconstrucción ($e_{\text{rec}} = (z_t - w_{\text{rec}}^T h_t)^2$, P10) y el error de pronóstico residual ($e_{\text{pred}} = (z_t - w_{\text{fc}}^T h_{t-1})^2$, P00) demostró dos naturalezas opuestas en el archivo UCR:

| Serie / Dataset | Longitud y Anomalía | P10 (Reconstrucción) | P00 (Pronóstico) | P05 (Híbrido) | Diagnóstico Físico |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`323_UCR_id_21_HumanActivity`** | 195,250 pts (1 pt anómalo en $t=128430$) | **0.0747** *(Invertido)* | **0.8605** *(Detección nítida)* | **0.7647** ($\Delta = +0.6900$) | **Ruptura de secuencia:** El estado actual $h_t$ absorbe el punto actual y anula el error reconstructivo, pero el estado previo $h_{t-1}$ no lo esperaba y estalla en pronóstico. |
| **`470_UCR_id_168_Medical`** | 41,790 pts (30 pts anómalos en $[10998, 11027]$) | **0.3093** *(Invertido)* | **0.2343** *(Invertido)* | **0.1660** ($\Delta = -0.1433$) | **Pérdida de dinámica (Amortiguación):** La señal se aplana. El error se va a cero en la anomalía y el detector puntúa más alto la zona normal. |

---

### 13.2 Descomposición Numérica Punto a Punto en `470_UCR` (41,790 puntos):
1. **Comportamiento de la señal cruda:**
   - **Zona Normal (41,760 puntos):** Media $= 1581.52$, Desviación estándar $\sigma = 178.53$, Rango $[1126.0, 2289.0]$.
   - **Zona Anómala (30 puntos, 0.07% de la serie):** Media $= 1685.33$, Desviación estándar $\sigma = 70.94$, Rango $[1514.3, 1763.0]$.
   - **Diagnóstico:** La anomalía está enteramente dentro del rango intercuartil normal, pero su variabilidad local se reduce a menos de la mitad ($\sigma$ cae de $178.5$ a $70.9$).
2. **Magnitud de los Errores Residuales del ESN:**
   - Error de reconstrucción $e_{\text{rec}}$: Media Normal $= 0.0150$ vs. Media Anómala $= \mathbf{0.0023}$ (**6.5 veces menor en la anomalía**).
   - Error de pronóstico $e_{\text{pred}}$: Media Normal $= 0.0377$ vs. Media Anómala $= \mathbf{0.0039}$ (**9.7 veces menor en la anomalía**).
3. **Distribución de Scores Percentiles:**
   - En P05, los puntos normales tienen mediana de score $0.4856$ (percentil 90 en $0.8947$, máximo $1.0000$).
   - Los 30 puntos anómalos tienen mediana de score $\mathbf{0.1488}$ (máximo $0.3923$).
   - **Cero de los 30 scores más altos** cayeron dentro de la ventana de anomalía. El 100% cayó en la oscilación normal.
   - El AUC-ROC resultante de $0.1660$ refleja que el **$83.4\%$ de los pares normal-anómalo están invertidos**.

---

### 13.3 Falsación Epistemológica: El Límite Teórico del Error Residual
1. **Punto Ciego Compartido:** El error residual $z - \hat{z}$ mide sorpresa respecto a la expectativa del modelo. Un tramo que pierde varianza y se vuelve regular es *extremadamente fácil de modelar y predecir* para un ESN.
2. **Inutilidad de Ensembles Lineales:** Combinar dos señales con el mismo punto ciego ($\alpha e_{\text{rec}} + (1-\alpha) e_{\text{pred}}$) no sólo no resuelve el problema, sino que lo agrava (P05 cae a $0.1660$ porque el pronóstico es aún más ciego ante la calma que la reconstrucción).
3. **Imposibilidad de Filtros Monótonos:** Ninguna función monótona no lineal ni calibrador de percentiles invierte un ranking donde $\text{Score}(\text{Normal}) > \text{Score}(\text{Anómalo})$.

---

### 13.4 Hipótesis de Detección Causal: Pérdida de Dinámica / Variabilidad Local
Para capturar anomalías donde la señal deja de variar sin generar un error residual elevado, se formula una señal ortogonal causal:
$$D_t = \left|\log \frac{\sigma_{\text{local}, t} + \epsilon}{\sigma_{\text{historia}, t} + \epsilon}\right|$$

**Protocolo de Verificación Estricto:**
1. Parámetros de prueba causal: Ventana local $W_{\text{loc}} = 10$ puntos, Referencia histórica $W_{\text{hist}} = 500$ puntos (excluyendo la ventana local para evitar fugas y permitir respuesta inmediata).
2. Probar $D_t$ de forma **aislada y autónoma** sobre el CSV real [`470_UCR_id_168_Medical_tr_10447_1st_10998.csv`](470_UCR_id_168_Medical_tr_10447_1st_10998.csv).
3. Prohibido combinar con $\max(P05, D_t)$ hasta no verificar primero si $D_t$ por sí solo discrimina la anomalía y eleva el AUC-ROC sobre las etiquetas reales sin distorsión de escala.

---

### 13.5 Medición Empírica de Dispersión y Persistencia Causal en `470_UCR`
Se ejecutó la prueba causal directa sobre los 41,281 puntos evaluados de la serie `470_UCR`:
- **$D_{\text{original}}$ (Instantáneo $W_{\text{loc}}=10$):** **AUC-ROC = 0.7349** (vs. **0.1660** en P05, ganancia de $\mathbf{+0.5689}$ en la ordenación global). Mediana normal $0.9131$ vs. anómala $1.2936$. Top-30: $0/30$.
- **$D_{\text{persistencia\_10}}$ (Media móvil causal $W=10$):** **AUC-ROC = 0.7286**, Top-30: $0/30$.

#### Autopsia Numérica de los Falsos Picos de Dispersión:
- **Naturaleza de la anomalía:** No es una línea recta plana ($\sigma=0$), sino una oscilación patológica amortiguada: $\sigma_{\text{local}} \approx 29.05$ (histórico $\approx 99.44$), lo que produce $D_t \approx 1.31$ (máximo $2.36$).
- **Pausas basales normales:** En la señal normal periódica, los intervalos isoeléctricos entre latidos presentan mesetas fisiológicas de 6–8 puntos con $\sigma_{\text{local}} \approx 2.41 - 4.32$. Al ser $\sigma_{\text{local}}$ tan cercano a cero, $|\log(\text{loc}/\text{hist})|$ escala a **$3.26 - 3.89$**, superando puntualmente el score de la anomalía.
- **Conclusión de persistencia:** La media móvil simple de 10 puntos no suprime los intervalos isoeléctricos (que duran ~8 puntos) e introduce un retraso de fase de borde que reduce levemente el AUC-ROC de $0.7349$ a $0.7286$. La dispersión instantánea captura la anomalía a nivel de distribución global (0.735 vs 0.166), pero requiere modelar la escala del ciclo o la forma para eliminar los descansos isoeléctricos basales.

---

### 13.6 Prueba y Falsación de la Transformación Asimétrica de Entropía ($S_{\text{shape}}$)
Se evaluó la hipótesis de penalizar ratios extremadamente pequeños mediante la función de entropía:
$$S_t = \begin{cases} -r_t \log(r_t) & \text{si } 0 < r_t < 1 \\ \log(r_t) & \text{si } r_t \ge 1 \end{cases}$$

- **Resultado empírico:** **AUC-ROC = 0.4898**, Top-30: $0/30$.
- **Causa matemática del colapso a nivel de azar:**
  1. La rama $r < 1$ posee una cota superior analítica estricta en $\max(-r \log r) = 1/e \approx 0.3679$.
  2. La señal oscilatoria periódica normal transiciona continuamente en cada latido por el rango $r \in [0.2, 0.5]$, haciendo que la mediana normal ($0.3529$) sea indistinguible de la mediana anómala ($0.3500$).
  3. En la rama $r \ge 1$ (sístole), $\log(r)$ alcanza valores de $0.7 - 1.1$, superando sistemáticamente el techo de la anomalía.
- **Veredicto:** **Falsada y descartada.** Se preserva únicamente la referencia empírica de $D_{\text{original}}$ ($0.7349$).

---

### 13.7 Prueba y Falsación de la Duración Causal Consecutiva de Baja Variabilidad
Se evaluó la hipótesis de que la duración temporal de la atenuación ($r_t < 0.35$) distinguiría la patología sostenida frente a las pausas fisiológicas breves:
- **Resultado empírico:** **AUC-ROC = 0.6471**, Top-30: $0/30$.
- **Datos brutos de rachas:**
  - **Duración máxima dentro de la anomalía:** **14 pasos** (la oscilación interna de la anomalía hace que $r_t$ fluctúe cruzando el umbral).
  - **Duración máxima en la señal normal:** **64 pasos** (los descansos basales fisiológicos se sostienen por más de 4 veces el tiempo de la anomalía).
- **Veredicto y Cierre:** La duración acumulada invierte aún más los picos extremos a favor de las pausas normales. **Se declara cerrada y archivada la familia de heurísticas basadas exclusivamente en dispersión escalar local (ventana, persistencia lineal y duración de racha).** No se agregarán capas heurísticas a P05.

---

## 14. Evaluación Diagnóstica de Novedad de Subsecuencias (Left_STAMPi / Matrix Profile Causal) (2026-10-19)

> **Criterio Epistemológico:** Se contrastó la hipótesis de error residual frente a la de novedad de subsecuencia. En lugar de preguntar *"¿qué tan difícil es predecir este valor?"*, se evalúa *"¿qué tan parecido es este patrón local a los patrones previos observados en el historial?"* mediante la implementación oficial causal `Left_STAMPi` de TSB-AD sobre STUMPY.

### 14.1 Resultados en la Serie Crítica `470_UCR_id_168_Medical`:
- **Parámetros:** Longitud $= 41,790$, Entrenamiento inicial $= 10,447$, Ventana de subsecuencia $W = 100$.
- **Tiempo de ejecución:** $92.03$ segundos.
- **Métricas:**
  - **Serie completa:** **AUC-ROC = 0.9995**, **Top-30 = 14 / 30** aciertos directos.
  - **Desde fin de entrenamiento (fuera de muestra):** **AUC-ROC = 0.9994**, **Top-30 = 14 / 30**.

| Detector / Señal | AUC-ROC en 470_UCR | Top-30 Anomalías | Diagnóstico del Mecanismo |
| :--- | :---: | :---: | :--- |
| **P10 (Reconstrucción ESN)** | 0.3093 | 0 / 30 | Invertido (la calma es fácil de reconstruir). |
| **P00 (Pronóstico ESN)** | 0.2343 | 0 / 30 | Invertido severo (la calma es predecible). |
| **P05 (Híbrido ESN titular)** | **0.1660** | **0 / 30** | Colapso total por punto ciego compartido. |
| **Dispersión local $D_t$** | 0.7349 | 0 / 30 | Discrimina globalmente pero falsea en descansos basales ($\sigma \approx 2$). |
| **Left_STAMPi (STUMPY Causal)** | **0.9995** | **14 / 30** | **Detección nítida casi perfecta.** La subsecuencia amortiguada no tiene análogo previo en el historial. |

### 14.2 Conclusión Teórica y Práctica:
1. **Validación del Mecanismo:** La atenuación patológica no es invisible por falta de capacidad del modelo, sino por la formulación de la pregunta. Para un ESN, una onda plana es ruido cero; para un comparador de subsecuencias 1-NN causal, una onda plana en una señal cardíaca es un patrón completamente anómalo y sin precedentes.
2. **Coste Computacional:** `Left_STAMPi` con memoria no acotada (`egress=False`) escala cuadráticamente con la longitud de la serie ($O(T^2)$). En series de ~40k puntos toma ~92s; en series de ~200k puntos (como 323) el cálculo intensivo de distancias requiere decenas de minutos salvo que se use un horizonte de memoria acotado.

---

### 14.3 Descomposición Crítica en 3 Dimensiones y Línea Hacia DAMP
Se aislaron las tres dimensiones del resultado empírico:

| Dimensión | Medición en `470_UCR` | Diagnóstico Ingenieril |
| :--- | :---: | :--- |
| **AUC-ROC** | **0.9995** | Separación global casi perfecta. Confirma la solidez del paradigma de subsecuencias. |
| **Top-30** | **14 de 30** | La priorización extrema aún contiene falsos picos por borde y alineación de ventana ($W=100$). |
| **Tiempo de cálculo** | **92.03 s** (41,790 pts) | **Inviable para producción sin poda:** Extrapolación cuadrática a 195k puntos supera los ~33 minutos. |

#### Veredicto Metodológico:
1. **No integración de Left_STAMPi:** El coste cuadrático de `Left_STAMPi` (con `egress=False`) impide su uso en streaming ligero. Acotar arbitrariamente con `egress=True` alteraría el horizonte causal sin justificación.
2. **Línea de Referencia Conceptual (DAMP):** El trabajo de discordias causales de Keogh et al. (DAMP) demuestra que el cuello de botella no es el Matrix Profile en sí, sino la búsqueda exhaustiva. DAMP implementa poda temprana casi lineal sin comparar contra todo el historial.
3. **Decisión Operativa:** No escribir un Matrix Profile propio ni abrir otra fase de micro-optimizaciones. Se deja correr la serie 323 como única prueba de contraste y se congela el chasis de producción intacto.