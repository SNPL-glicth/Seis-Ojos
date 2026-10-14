# Predictor de contexto variable (PPM, D=6) — FALLIDA (2026-10-14)

## Resultado
NAB: **18.63 (−26.56)** | Low FP 5.48 | Low FN 29.09. Pierde en las 7
categorías (Tweets −13.46, AWS −12.92, Traffic −11.84). 7 archivos +1.

## Lo que sí funcionó
Trie acotado (halving+poda, nodos ≤4096 verificados), retroceso por
conteo, O(D)/paso a 0.03ms/pt, invarianza reset, causalidad. El test del
triple inédito demuestra capacidad de memoria de secuencias en sintético.

## Autopsia
En ruido real la sorpresa no separa (max_win ≈ p99_fuera en barato). La
cuantización escalar destruye la forma antes de la memoria: da igual el
orden del contexto si los símbolos no preservan similitud. HTM predice
representaciones densas, no símbolos.

## Lección
Representación densa primero, memoria después. Requiere embeddings
continuos = otro proyecto. Archivado. Configuración oficial vigente (45.19).

## Artefactos
- `experimental/context_predictor.py`, `tests/test_context_predictor.py` (7)
- `archive/gate_contexto_variable.py`, `results/sixeyes_contexto_variable.json`
