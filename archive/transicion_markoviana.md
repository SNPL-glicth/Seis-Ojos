# Transición markoviana K=8 — FALLIDA (2026-10-14, archivado)

## Resultado
NAB standalone: **14.36 (−30.83)**, bajo `random` (16.83). Pierde en las 7
categorías. 7 archivos +1 vs 27 −1.

## Hallazgos en implementación (valiosos, independientes del score)
1. Off-by-one en la especificación: 6 cotas → 7 regiones (símbolo 6
   inalcanzable). Corregido con colas ±3.
2. Intuición del flatline falsada con números: congelamiento tras ruido =
   1.45 bits vs mediana 2.69 del régimen. S→S central es común, no raro.
3. MAD=0 enmudecía flatlines (rogue 0.00): dispersión cero + desvío ahora
   es símbolo extremo (rogue → 5.21 bits).

## Autopsia del colapso
Sorpresa alta en todo ruido (mediana ~2, p99 ~5–6 dentro/fuera): sin
separación, el umbral global solo deja picos → FNs masivos. Orden 1 sobre
z puntual no es "ley de transición": falta memoria de secuencias. La vía
vive solo como predictor de próximo símbolo, no como scorer.

## Criterio de aceptación
- Aislamiento ✓ (`experimental/markov_transition.py`), invarianza ✓,
  causalidad ✓.
- Sorpresa alta en ciegas: parcial (3/4; rogue necesitó fix, sigue bajo
  p99 externo).
- LOCO sin regresiones: NO alcanzado (−30.83 global). Archivado.

## Artefactos
- `experimental/markov_transition.py`, `tests/test_markov.py` (7)
- `archive/gate_markov.py`, `results/sixeyes_markov_orden1.json`
- Configuración oficial vigente (dualscale, 45.19).
