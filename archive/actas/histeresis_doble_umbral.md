# Histéresis θ_high=0.998/θ_low=0.95 — FALLIDA (2026-10-14)

## Resultado
NAB: **41.77 (−3.42)** | Low FP 31.37 (+2.24 vs 29.13) | Low FN 48.54 (+1.74).
Gate barato previo: nyc_taxi TP 10→24 (FP 21→29).

## Por categoría (standard)
Tweets −4.37, KnownCause −2.53, resto ≈0 (AWS 0.00 exacto). Ningún archivo
gana >1; 2 pierden >1.

## Autopsia
El latch sostiene cola: en ráfagas sostiene FPs tanto como TPs. PeakDecay
atenúa cola y gana en este corpus. La forma correcta de la cola es atenuar.

## No-ops verificados (no implementados, con causa)
- Silencio probatorio 15%: Δ=0 por construcción (scorer excluye el tramo;
  MIN_HISTORIAL ya emite 0.0 al inicio).
- Umbral manual 0.995: redundante (el optimizador elige el óptimo por perfil
  sobre scores continuos; no hay umbral propio que bajar).

## Artefactos
- `experimental/hysteresis.py`, `tests/test_hysteresis.py` (5)
- `archive/gate_histeresis.py`, `archive/diff_histeresis.py`
- `results/sixeyes_histeresis.json`
- Configuración oficial vigente (dualscale, 45.19).
