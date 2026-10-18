# Six Eyes — Biblia Técnica y Forense (`system.md`)

> Documento maestro de arquitectura. Fuente de verdad de ingeniería: qué es el sistema,
> por qué gana, por qué todo lo demás pierde, y por qué el benchmark no deja subir más.
> Números con artefacto en `results/` o no existen. Referencia viva: `README.md` (titular),
> `aprendizaje.md` (registro cronológico), `run_nab.py` (fábrica canónica).

**Puntajes canónicos (no editar sin re-medir 3 semillas + LOCO):**

| Medición | Valor | Artefacto |
| :--- | :---: | :--- |
| Media 3 semillas in-sample (titular) | **57.68 ± 3.08** | `sixeyes_refractory_w80_t990` ×3 seeds |
| Mejor semilla (seed 42, histórico) | 60.81 | `results/sixeyes_refractory_w80_t990.json` |
| LOCO 6 folds out-of-sample | **55.72 ± 10.52** | `loco_sixeyes_abl_full_s{42,7,99}.json` |
| Ensemble determinista + LOCO (verídico) | 54.45 ± 13.37 | variante `ensemble_rangos` |
| Base ESN sin capas | 48.30 ± 1.44 | `sixeyes_esn_k24_tau2p0_w16000.json` |

---

## 1. Mapa del Código Base (`src/sixeyes/`)

Todo el sistema es streaming causal estricto: `update(value, timestamp) -> float in [0,1]`,
solo pasado + punto actual, memoria acotada, `reset() + replay == replay`.
Complejidad citada como coste por punto; toda constante dimensional (Nres, K, W)
está fijada a priori, luego todo es **O(1) respecto a T** (tiempo de serie).

### 1.1 `detectors/` — Núcleo de producción (único código que puntúa en NAB oficial)

| Módulo | Clase / Función exacta | Rol en el pipeline | Complejidad / punto |
| :--- | :--- | :--- | :--- |
| `base.py` | `Detector` (Protocol runtime_checkable) | Contrato: `update(value, timestamp)->score`. Todo detector/envolvente lo implementa. Permite anidar envolventes como decoradores. | — (tipo) |
| `delay_embedding.py` | `DelayCoordinateEmbedding(d=16, stride=1)` | Incrustación de Takens: escalar `x_t -> [x_t,...,x_{t-15}] ∈ R^16`. Buffer circular `len=(L-1)*tau+1`. Sin diferencias finitas (no amplifica varianza ×6). | O(L)=O(16) tiempo, O(L) memoria → **O(1)** |
| `echo_state.py` | `EchoStatePredictor(d=16, Nres=64, ρ=0.9, λ=0.999, seed)` | **Predictor oficial.** Estandariza causal (W=6400, sumas O(1)) → `h_new=tanh(W_in·vec + W_res·h)` → `pred=w_out·h_new` → `err=(z-pred)²`. Actualiza `w_out` por RLS **después** de puntuar. `W_in, W_res` fijas aleatorias, radio espectral 0.9 (propiedad de eco). | O(Nres²)=4096 flops (mat-vec + RLS) → **O(1)**. ~0.07 ms/pt |
| `calibrador_percentil.py` | `PercentileCalibrator(detector, max_history=16000)` | **ECDF causal acotada.** Convierte error crudo no-calibrado a percentil `score=(menores+0.5·iguales+0.5)/(n+1)` vía `bisect` sobre historial ordenado + cola FIFO. `MIN_HISTORIAL=288` (devuelve 0.0 antes). W=16000 cubre p90 de NAB (max 22695). | Inserción O(log W)+desplazamiento O(W) acotado por W fijo → **O(1)** amortizado |
| `peak_decay.py` | `StreamingPeakDecay(detector, K=24, τ=2.0)` | **NMS causal + decaimiento.** Si `raw > max(hist[K])` emite cresta; si no, `raw·exp(-(t-t_pico)/τ)`. Pasa intacto el frente de ataque, atenúa mesetas/réplicas. Inocente verificado 3× (`peak==cal` en atribución). | O(K)=O(24) `max` sobre deque → **O(1)** |
| `spatial_spike_bypass.py` | `SpatialSpikeBypass(detector, tol=0.05, warmup=100)` | **Bypass de ruptura de rango.** Mantiene `[min,max]` causales. Si `x_t > max+margen` o `< min-margen` con `margen=span·0.05` tras 100 pts, emite `1.0` directo sin pasar por predictor. Re-ancla cotas cada paso. `span≈0` jamás dispara. | O(1) tiempo y memoria |
| `refractory_suppressor.py` | `RefractorySuppressor(detector, w_rest=80, trigger=0.990)` | **Silencio causal post-pico.** Si `raw>=0.990`, silencia a `0.0` los siguientes 80 pasos (`silence_until=t+1+W`). Explotación directa de la asimetría NAB (TP=MAX, FP=aditivo). | O(1) tiempo y memoria |
| `dual_scale.py` | `SeasonalDualScaleRobustZ / DualScaleRobustZ` | Baseline denso alternativo (no oficial): `max(z_rápida, z_lenta)`. Segunda base medida 48.32. | O(1) |
| `seasonal.py` / `biseasonal.py` | `DailySeasonalRobustZ / BiSeasonalRobustZ` | Baselines por bucket temporal (15 min). Históricos 40.02 / variantes. | O(1) |
| `zscore_robusto.py` | `RollingRobustZ` | Baseline mediana/MAD rodante. | O(W) acotado → O(1) |
| `subespacio.py` | `SubspaceResidualDetector` | Baseline de residuo de subespacio. | O(1) |
| `aleatorio.py` | `RandomDetector(seed=42)` | Nulo determinista para sanity. | O(1) |
| `combinador.py` / `ramp_gradient.py` | Utilidades de fusión / gradiente | Soporte experimental, fuera del pipeline oficial. | O(1) |

Orden canónico de anidado (fábrica `build_factory("refractory")` en `run_nab.py:76-84`):

```
EchoStatePredictor(seed)
 → PercentileCalibrator(max_history=16000)
  → StreamingPeakDecay(k=24, tau=2.0)
   → SpatialSpikeBypass(tol=0.05, warmup=100)   # score 1.0 si rompe rango
    → RefractorySuppressor(w_rest=80, trigger=0.990)  # silencio causal
```

Coste total medido: **~0.11 ms/punto (~9k pts/s monohilo)**. Apto para muestreos de minutos.

### 1.2 `dense_arch/` — Representación densa alternativa (segunda base, no oficial)

| Módulo | Rol | Complejidad |
| :--- | :--- | :--- |
| `encoder.py` (`PhaseSpaceEncoder`, d=16 → R^32, proyección gaussiana fija JL, seed 42) | Proyección Johnson-Lindenstrauss sin cuantización. Preserva geometría en flotante. | O(m·d)=512 → O(1) |
| `trajectory.py` (`DualTrajectoryNovelty` / `TrajectoryNovelty`, γ=0.98/0.999) | Distancia al centroide EWMA rápido/lento, `score=max(ambas)`. NAB 48.32 (−0.73, en ruido), LOCO 44.44. Queda como base 10× más barata (0.011 ms/pt). | O(m) → O(1) |

### 1.3 `transition/` — Motor de transición STE (archivado, §3.5)

| Módulo | Rol | Complejidad |
| :--- | :--- | :--- |
| `reservoir_state.py` (`ReservoirState`) | Réplica exacta de la dinámica del reservorio ESN (mismas matrices/semilla) **sin** readout RLS. Expone `h_t ∈ R^64`. | O(Nres²) → O(1) |
| `state_quantizer.py` (`OnlinePrototypeQuantizer`, K=32, lr=0.02) | Mini-k-means streaming: asigna al prototipo más cercano **antes** de adaptar; init determinista primeros K estados. | O(K·d) → O(1) |
| `transition_model.py` (`TransitionModel`, ε=1.0) | Markov orden-1 `P(j\|i)=(C[i,j]+ε)/(fila+Kε)`. `observe` después de puntuar. | O(K²)=1024 memoria → O(1) |
| `transition_scorer.py` (`TransitionSurpriseDetector`) | `score=-log P(S_t\|S_{t-1})`. Orquesta reservorio→cuantizador→matriz en orden causal. | O(1) |

### 1.4 `evaluation/` — Medición honesta

| Módulo | Rol |
| :--- | :--- |
| `nab_runner.py` (`NABRunner`) | Envuelve `Sweeper` oficial NAB. Genera CSVs de detección por serie, barre umbral global, escribe `results/sixeyes_<config>.json` aislado. Jamás muta `final_results.json`. |
| `nab_cv.py` | **LOCO real por categoría** (7 cats → 6 folds con ventanas; `artificialNoAnomaly` excluida por `perfect=0`). Detecciones generadas UNA vez; umbral barrido SOLO en folds de entrenamiento y transferido al held-out. Normalización `100·(raw-null)/(perfect-null)` por fold. |
| `tsb_runner.py` | Puerta TSB-Eva (472 series). `statsmodels+sklearn` instalados 2026-10-15, suite en verde. Candidato congelado a correr UNA vez. |

### 1.5 `experimental/` — Cementerio documentado (ninguno puntúa en oficial)

`bocpd.py`, `level_rarity.py`, `persistence_filter.py`, `contextual_likelihood.py`,
`state_volatility.py`, `markov_transition.py`, `multi_horizon_esn.py`, `adaptive_tail.py`,
`asymmetric_refractory.py`, `hybrid_orthogonal.py`, `rank_ensemble.py`, `knn_detector.py`,
`gram_tracker.py`, `causal_spectral.py`, `causal_multiscale_lds.py`, `global_norm_esn.py`,
entre otros. Cada uno con `tests/test_<x>.py` en verde + gate en `archive/gates/` +
artefacto en `results/` + entrada de autopsia en `aprendizaje.md`. Ver §3.

---

## 2. Anatomía del Pipeline Oficial (El Éxito)

### 2.1 Física y matemática de la combinación ganadora

**Capa 1 — `EchoStatePredictor`: lectura RLS (memoria de lo normal).**
Entrada estandarizada causalmente `z_t=(x_t-μ_{6400})/σ_{6400}` (sumas rodantes O(1)).
Vector de Takens `v_t ∈ R^16`, estado `h_t=tanh(W_in·v_t + W_res·h_{t-1}) ∈ R^64` con
`W_res` reescalada a radio espectral ρ=0.9 (eco: memoria ~1/(1-ρ) ≈ 10 pasos efectivos,
suficiente para dinámica local sin inestabilidad). Readout lineal `ŵ` adaptado por
RLS con olvido λ=0.999 (`P₀=I`, `K=Ph/(λ+hᵀPh)`, `ŵ+=K·(z-pred)`). Score crudo
`e_t=(z_t-ŵᵀh_t)² ≥ 0`. Crucial: **predice con el pasado, puntúa, y DESPUÉS aprende** —
cero fuga. Lo único aprendido del sistema es `ŵ`: la referencia de "qué es normal en t+1".
Sin `ŵ`, el reservorio es proyección aleatoria fija (ver §3.4).

**Capa 2 — `PercentileCalibrator` + `StreamingPeakDecay`: del error a la cola extrema.**
El error cuadrático tiene skew 34 / kurtosis 1402: incomparable entre series.
La ECDF causal `F̂(e_t)` lo mapea a percentil uniforme en [0,1]. Luego PeakDecay
aplica NMS causal: conserva solo nuevas crestas locales en ventana K=24 y decae el
resto `exp(-Δt/2.0)`. Elimina la cola de relajación sin tocar el frente de ataque.
Micro-grid in-sample declarado: K=24 (49.05) > K=16 (48.72) > K=12 (48.62) > K=8 (48.38);
τ irrelevante (bit-idéntico en {1.5,2.0,2.5}).

**Capa 3 — `SpatialSpikeBypass`: rupturas de rango (el verdadero motor de NAB).**
`[min,max]` causales + `margen=0.05·span`. Ruptura → `1.0` inmediato, sorteando la
latencia de asimilación del embedding (16 pts) + RLS. Ablación pareada 3 semillas
(replay offline bit-a-bit ~1e-16): **spatial−base +10.20 (3/3 positivo, ±0.62)** —
el componente robusto. Dato demoledor: **regla trivial+bypass 50.83 > ESN entero 49.05**;
dado el bypass, ESN aporta +0.28 sobre doble-escala (58.50 vs 58.22). **NAB lo mueven
las rupturas de rango, no el predictor.**

**Capa 4 — `RefractorySuppressor`: silencio causal (arbitraje de la función de coste).**
NAB premia TP con operador MAX (primer impacto en ventana = crédito completo, resto = 0)
pero penaliza FP punto-a-punto aditivamente (−0.11 c/u en Standard; −0.20 aprox. en LowFP).
El ESN emite ráfagas de 3–5 alertas por evento que se desbordan fuera de la ventana oficial,
forzando al optimizador a subir el umbral global a 0.99658 y cegando series sutiles.
El refractario (disparo ≥0.990 → silencio 80 pasos) amputa las réplicas turbulentas,
permitiendo que el **umbral óptimo global descienda a 0.99077** y rescatando TPs en
todo el corpus. Auditoría de silencio: 649/647/663 silencios, 214 bypass-hits
independientes de semilla, 16–19/116 ventanas empiezan en silencio (~11k/33.5k puntos
cubiertos) — coste contado, neto positivo.

### 2.2 Por qué extrae 57.68 ± 3.08 y LOCO 55.72

Descomposición del 60.81 (seed 42):

| Pieza | Puntos |
| :--- | :---: |
| Regla de rango sola (trivial + bypass) | 50.83 |
| Cualquier predictor encima | +7.5 |
| Suerte de semilla 42 | +0.30 |
| Refractario | +2.01 (solo seed 42) |
| **Total** | **60.81** |

Multisemilla (42/7/99): 60.81 / 54.66 / 57.58 → **media 57.68 ± 3.08** (vs base 48.30 ± 1.44:
**+9.38 neto**). Por categoría (seed 42): AWS −2.21→+6.14, AdExchange +0.09→+5.13,
KnownCause −0.93→+3.68, Tweets +0.61→+6.11, ArtWith −4.14→−0.38, Traffic +4.37→+5.07.
Las 6 categorías reales ganan masiva y consistentemente.

LOCO pareado por semilla (full vs spatial-solo):

| Semilla | Full | Spatial-solo | Δ |
| :---: | :---: | :---: | :---: |
| 42 | 60.06 | 48.87 | +11.19 |
| 7 | 50.33 | 50.78 | −0.45 |
| 99 | 56.76 | 51.16 | +5.60 |
| **Media** | **55.72** | **50.27** | **+5.45 ± 3.36** |

El refractario **baja** la media in-sample (57.68 vs 58.50 spatial-solo) pero **sube** la
transferencia (+5.45 pareado, 2/3 positivo). Su valor está en el **umbral transferido**
(0.99077 vs 0.99658), no en el score. Folds LOCO Standard: 60.2 / 68.3 / 59.7 / 68.1 /
57.3 / 46.8; desviación 15.64→7.26. Veredicto: empate técnico con Skyline (58.20) y
KNN-CAD (57.99) dentro del ruido ±3.08. Puntaje verídico determinista: **54.45**.

---

## 3. Catálogo Forense de Autopsias (Los Fracasos Archivados)

Criterio pre-registrado de promoción (inmutable): **media 3 semillas ≥ 60.68 (+3 sobre
57.68), ninguna semilla peor, sin regreso en LOCO**. Todo lo abajo está medido en NAB
completo con artefacto y archivado por incumplirlo. Ningún número sin `results/`.

### 3.1 BOCPD — Falsado por latencia y sobreajuste de distribución (36.59)

- **Mecanismo intentado:** Adams & MacKay (2007), NIGamma conjugado online
  (`μ₀=0, κ₀=0.5, α₀=1, β₀=1`), hazard `h=1/250`, truncamiento `max_rl=500`,
  estandarización causal W=500. Score = sorpresa predictiva marginal
  `−log P(x_t|x_{1:t-1})` (mixture Student-t sobre run-lengths).
  `src/sixeyes/experimental/bocpd.py`, 9 tests, `results/sixeyes_bocpd.json`.
- **Resultado métrico:** **Standard 36.59 (−12.46 bajo ESN 49.05)** | LowFP 25.39 |
  LowFN 42.71. TPs: 84 vs 213 del ESN (−60%). Único brillo focal: `realTweets` +3.10.
- **Causa raíz exacta:** el prior NIGamma asume observaciones **i.i.d. dentro de cada
  régimen**. Ante estacionalidad periódica o tendencias suaves (AWS −9.92, KnownCause
  −11.58, Traffic −8.15), cada cresta normal infla la predictiva o dispara changepoints
  espurios. Sin modelo dinámico interno (memoria de secuencia/recurrencia), el modelo
  probabilístico i.i.d. no compite con el reservorio en series cíclicas/industriales.
  **Latencia estructural:** necesita acumular evidencia para declarar régimen nuevo;
  el ESN puntúa el error instantáneo. Línea standalone archivada.

### 3.2 LevelRarity & Filtro de Persistencia — Falsados por inundación de FPs y dispersión de impulsos (55.50 → 53.06)

- **Mecanismo intentado (LevelRarity):** generalización del bypass ("salió del rango" →
  "está en zona rara aunque dentro del rango"). Histograma K=80 bins sobre rango
  warmup(100)+5% margen+2 overflow, semivida 2000, score = p-valor de cola
  `1−masa(bins ≥ tan raros)/total` calculado ANTES de insertar el punto.
  `src/sixeyes/experimental/level_rarity.py`, 7 tests, `results/sixeyes_level_rarity.json`.
- **Resultado métrico:** **Standard 55.50 (−2.18 bajo 57.68)** | LowFP 38.40 | LowFN 62.90.
  Mecanismo funcional: **14/17 ventanas de la brecha recuperadas** (grok 0.9991,
  machine_temp 0.9985/0.9927, nyc_taxi 0.9972/0.9914, TravelTime 1.0/0.9923, etc.).
  Pero **418 FPs nuevos** fuera de ventana destruyen el neto.
- **Mecanismo intentado (PersistenceFilter M=2):** hipótesis "los FPs son impulsos de
  1 paso; exigir M=2 consecutivos en anomalía los elimina sin sacrificar TPs".
  `src/sixeyes/experimental/persistence_filter.py`.
- **Resultado métrico:** **Standard 53.06 (−4.62 vs full, −2.44 vs LevelRarity)**.
  FPs **suben de 418 a 726**. Las 14 ventanas se mantienen pero el score colapsa.
- **Causa raíz exacta:** los 418 FPs son **418 episodios de 1 punto, 0% permanencia**
  (`archive/gates/diag_rarity_fps.py`, sin corrida nueva): cruces aislados y dispersos
  del umbral en colas de densidad, no permanencia en zona rara. El refractario (80 pasos)
  ya silencia toda permanencia; **no hay dimensión entrada-vs-permanencia que recortar**.
  Al alterar la dinámica temporal del refractario+umbral, el filtro multiplica los cruces
  efectivos. Cuenta del revisor: 14 ventanas ~+11, 418 FPs ~−13, neto ~−2; para +3 harían
  falta −40% FPs conservando aciertos, imposible siendo aislados. Familia histograma
  confirmada en su techo: relativeEntropy (no-causal, binaria, ajustada) 54.64 con
  17/17 y 373 FPs vs nuestro causal 55.50 con 14/17 y 418 FPs — **techo de familia ~55**.

### 3.3 Contextual Anomaly Likelihood sobre Residuos — Falsado por saturación de la ventana de varianza en regímenes mixtos (58.02 ± 0.94)

- **Mecanismo intentado:** Numenta Likelihood (Ahmad et al. 2017) + ContextOSE sobre error
  ESN crudo. EMAs O(1): `μ_short`(W′=10), `μ_long, σ_long`(W=1000); z-score
  `(μ_short−μ_long)/σ_long` → `score=Φ(z)=0.5·(1+erf(z/√2))`.
  `src/sixeyes/experimental/contextual_likelihood.py`, 5 tests.
- **Resultado métrico (3 semillas pareadas sobre `raw_esn`):** s42 **57.79 (−3.02)** |
  s7 **57.23 (+2.57)** | s99 **59.04 (+1.46)** → **media 58.02 ± 0.94 (+0.34 neto,
  lejos del +3.0 exigido)**. LowFP +6~8 pts, LowFN −5~7 pts.
- **Causa raíz exacta (Paradoja de Adaptación Local):** en series estables reduce FPs
  espectacularmente (artificialNoAnomaly 9→3, LowFP 43.40→~50.5, realTweets +5.58).
  Pero en series largas con regímenes mixtos (`machine_temperature`, `nyc_taxi`), un
  error moderado sostenido es absorbido por W=1000: **infla σ_long y desplaza μ_long**,
  normalizando como "esperado" las anomalías posteriores de la misma serie. TPs del
  corpus 147→98. Colapso: KnownCause +3.68→−2.90 (−6.57), Traffic +5.07→+0.14 (−4.92).
  La adaptatividad local canibaliza la señal que pretende rescatar. Archivada con evidencia.

### 3.4 Espacio Latente del Reservorio (`h_t ∈ R^64`) — Falsado por contracción geométrica y colapso a puntos fijos (21.61)

- **Mecanismo intentado (`StateVolatilityDetector`, `Δh=‖h_t−h_{t-1}‖₂`):** usar la
  trayectoria interna completa R^64 en vez del error escalar R^1 del readout, para ganar
  dimensionalidad (perturbaciones en cualquier dirección) y latencia (pesos fijos, sin
  olvido RLS que amortigüe). Mismo reservorio/semilla, RLS mantenido "en forma" aunque
  no puntúe. `src/sixeyes/experimental/state_volatility.py`, 9 tests,
  `results/sixeyes_state_volatility.json`.
- **Resultado métrico:** **Standard 21.61 (−27.44)** | LowFP 11.86 | LowFN 28.41.
  Score máximo del optimizador **−65.87** (casi ruido puro).
- **Causa raíz exacta (doble):**
  1. **Sin referencia aprendida no hay discriminación.** El reservorio es proyección
     aleatoria fija; lo único aprendido es `ŵ` (RLS). Medir `‖Δh‖` descarta exactamente
     el clasificador y conserva el extractor aleatorio: `Δh` es alto ante TODA entrada
     volátil (anómala o no) porque el estado siempre cambia. Análogo a medir activaciones
     intermedias de red aleatoria sin cabeza clasificadora.
  2. **Geometría adversa verificada:** en caídas de amplitud, `tanh(W·v+W_res·h)` se
     **contrae hacia el origen** (entradas pequeñas → estado pequeño → `‖Δh‖` pequeño
     justo cuando debería gritar); en flatlines, el reservorio **colapsa a punto fijo**
     (`h*=tanh(W·c·1+W_res·h*)`) y `‖Δh‖→0` aunque el flatline sea la anomalía.
     La norma geométrica es anti-señal en los dos arquetipos NAB. Línea cerrada.

### 3.5 State Transition Engine (STE / Markov) — Falsado porque ruido y anomalías generan sorpresas equivalentes bajo umbral global (37.04)

- **Mecanismo intentado (3 generaciones):**
  - (a) `MarkovTransitionSurprise` escalar: K=8 símbolos por cotas fijas sobre z robusto,
    matriz K×K con olvido λ=0.999, `score=−log₂ P̂(S_t|S_{t-1})`, Laplace ε=1.0 → **14.36**.
  - (b) PPM de contexto variable (símbolos pobres) → **18.63**.
  - (c) `TransitionSurpriseDetector` (STE): estados densos R^64 + cuantizador online K=32
    (lr=0.02, init determinista) + Markov orden-1 + `score=−log P`. 8 tests, todo a priori
    sin barrido → **media 37.04** (42=39.43, 7=35.22, 99=36.48; las 3 peores que su full).
- **Causa raíz exacta:** la representación densa SÍ ayuda (37.04 > 14.36 > 18.63 > 21.61
  parcial) pero pierde contra error-de-predicción ESN (48.30). La sorpresa de transición
  es alta **en regímenes volátiles normales igual que en anomalías**: sin la referencia
  aprendida del readout ("qué es normal"), no hay separación bajo **umbral global único**.
  El ruido de cola y la anomalía generan distribuciones de sorpresa solapadas; el
  optimizador no encuentra umbral que las parta. Tercera falsación transicional cierra la
  familia salvo modelo de segundo orden/jerárquico (otro proyecto). Artefactos
  `results/sixeyes_ste_s{42,7,99}.json`.

### 3.6 Rama Periódica Residual (`PeriodicResidualDetector` + MaxFusion) — Empate dentro del ruido, no cumple el criterio de promoción (58.02 ± 2.40)

- **Mecanismo intentado (`src/sixeyes/experimental/periodic_residual.py`):** rama
  paralela que modela dependencia con lags autoperiódicos sobre el valor crudo,
  calibrada causalmente (`PercentileCalibrator`, W=16000) y fusionada por máx con la
  rama ESN cruda (`MaxFusion`). Cadena aguas abajo idéntica al baseline: PeakDecay
  (K=24, τ=2), SpatialSpikeBypass, RefractorySuppressor (w=80, t=0.990).
  Auditoría causal en `scripts/experiments/eval_periodic_nab.py`, 3 semillas pareadas
  sobre `raw_esn`, `results/sixeyes_periodic_fusion_s{42,7,99}.json`.
- **Resultado métrico (pareado por semilla contra su baseline correspondiente):**
  s42 **61.16 (+0.35)** | s7 **55.34 (+0.67)** | s99 **57.55 (−0.03)** → **media 58.02
  ± 2.40 vs 57.68 ± 3.08: diferencia media +0.33**. Contra el pareado multisemilla
  (60.81 / 54.66 / 57.58), la fusión **empata dentro del ruido, no pierde**; pero
  **no cumple el criterio pre-registrado** (media ≥ 60.68, ninguna semilla peor): la
  media queda a −2.66 y ninguna semilla supera a su baseline de forma significativa.
- **Causa raíz exacta (asimetría rescate/supresión):** en s42 no pierde ventanas
  (4 ganadas: 3 artificiales + `ec2_cpu_utilization_53ea38`) pero la fusión por máx
  **infla la tasa de activación upstream**, con coste puro de FPs reales:
  376→402 (+26), KnownCause 71→87, `cpu_utilization_asg_misconfig` +16
  (21→37, sin ventanas nuevas) → Delta FP total +45 contra Delta H +4
  (cociente marginal 11.25, bajo el equilibrio ~17.01 solo porque el rescate se
  concentra en ventanas baratas de artificiales). En s7/s99 el patrón invierte:
  compuerta abierta en ~90% de las series desplaza el refractario y **suprime
  ventanas valiosas** (nyc_taxi 7→2 TP, Twitter_volume_AAPL 9→6 TP,
  `asg_misconfig` 10→9 TP con +19 FP). Sin compuerta selectiva por periodicidad
  real (la rama abre en series apériodicas), el signo del intercambio depende de
  la semilla. Archivada.

**Falsaciones laterales que confirman el diagnóstico (no promocionan):**

**Falsaciones laterales que confirman el diagnóstico (no promocionan):**
multi-horizonte [1,3,6] 43.42 (ruido de base en h=3,6 domina la norma); likelihood
L1/L2/L3+firmada 33.47/30.11/39.51/32.89 (likelihood corta-vs-larga no discrimina);
fusión ortogonal ESN+centroide 42.05 (13+ operadores de fusión, ninguno bate al single);
`time_features=True` 47.43/56.55 (pesos fijos+entradas periódicas=ruido estructurado);
refractario asimétrico 59.32 (wakeups fragmentan silencio, umbral sube a 0.99649);
cola adaptativa por serie 1.45 (error de signo: cola superior P(X≥s) pequeña=anómala,
señal invertida); k-NN disperso 26.92/28.58; warmup/winsorización revertidos.

---

## 4. Límites Estructurales del Benchmark

### 4.1 La función de coste impone el techo, no la falta de ideas

NAB Standard: cada FP −0.11 (LowFP ≈ −0.20), cada TP +crédito una sola vez vía MAX en
ventana, con decaimiento por distancia al borde y probation del 15% inicial ignorado.
**Un único umbral global** sobre 58 series heterogéneas (AWS, AdExchange, KnownCause,
Traffic, Tweets, Artificial). El `Sweeper` oficial ya encuentra el mejor tradeoff TP/FP
global. Consecuencias medidas:

1. **Toda señal nueva paga peaje global.** Disparar en cualquier serie suma FPs a
   −0.11 cada uno en las 58 a la vez. Para ganar neto debe aportar TPs que valgan más
   que sus FPs. Nada probado lo logra: las 17 ventanas de la brecha valen ~+11 pero
   cuestan ~+13 en FPs capturarlas (LevelRarity: 418 FPs aislados).
2. **Bajar el umbral rescata ventanas pero admite ruido corpus-wide.** Las 17 ventanas
   perdidas (nyc_taxi×2, grok, TravelTime×2, machine_temp×2, exchange-3/4_cpc, etc.,
   scores 0.92–0.99 vs umbral 0.99077/0.99658) son subidas moderadas DENTRO del rango:
   invisibles para rareza estática `P(anomalía|x_t)`, solo visibles para transición
   `P(anomalía|x_t,...,x_{t-k})` — familia que no separa bajo umbral global (§3.5).
3. **El refractario estático funciona PRECISEMENTE porque no perdona.** Su rigidez
   permite el umbral bajo 0.99077; cualquier flexibilización (asimétrico con wakeups,
   persistencia, likelihood adaptativa) fragmenta el silencio, el optimizador sube el
   umbral (0.99649) y el neto cae. Refutación aritmética: contextOSE gana con MENOS TPs
   (96<147) y −318 FPs — el problema son los 382 FPs, no las 17 ventanas; pero recortar
   a ~180 FPs sin perder TPs no se logró en 6+ intentos.
4. **Cota por-archivo 75.71 vs 49.05 global (+27).** La señal CONTIENE la anomalía
   (percentil mediano de máximos en perdidas: 99.68); el cuello es calibración+umbral
   global, no el detector. Controles: Spearman reescalado 1.0000 (nada que hacer),
   R² umbral-vs-propiedades ≤0.05 (impredecible sin etiquetas), AUC INVALIDADO como
   selector (k-NN 0.864↔35.16). Los +27 son inalcanzables causalmente sin etiquetas.

### 4.2 Techo certificado y única vía real

Seis+ intentos archivados convergen al mismo techo (~55–58): LevelRarity 55.50,
PersistenceFilter 53.06, relativeEntropy no-causal ajustada 54.64, ContextualLikelihood
58.02 (+0.34, n.s.), ensemble 54.45 LOCO, STE 37.04, BOCPD 36.59. Contaminación
declarada: ≥37 configs sobre las mismas 58 series (W×t + grid K×τ + tol/warmup) —
todo número in-sample es segundo in-sample.

**La única vía real es TSB-Eva (350 series intocadas + 822 Eva-Full, datos ya en disco
verificado 350/350 y 822/822).** Si el candidato congelado
(ESN+ECDF16000+PeakDecay24/2.0+bypass5%/100+refractario80/0.990) generaliza fuera de
NAB, es señal real; si no, **54.45 es el techo honesto**. NAB queda cerrado:
titular 57.68, LOCO 55.72, techo documentado.

---

*Fin de `system.md`. Editar solo con medición pareada + artefacto + entrada en
`aprendizaje.md`. Prohibido: reglas por archivo/categoría, umbrales por nombre de serie,
traducción línea-por-línea de código AGPL (ARTimeNAB.jl — solo ideas desde diseño propio).*
