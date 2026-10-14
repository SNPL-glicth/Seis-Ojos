# Período adaptativo — FALLIDA (archivada 2026-10-14, archivado)

## Hipótesis
Los 96 buckets fijos de 15 min dejan 72 buckets muertos en series horarias
(7/58) y desalinean el período en 1min/30min (3/58). Estimar `dt` causalmente
y colapsar a m=1440/dt buckets debería ganar en `realAdExchange` y
`realKnownCause` sin tocar las 48 series de 5 min.

## Medición (NAB completo, K=12, tau=2.0, W=16000)
- Baseline `results/sixeyes_peak_decay_b15_k12_tau2p0_w16000.json`: **40.02**
- Variante warmup 200 pts): `results/sixeyes_periodo_adaptativo_warmup200.json`: **36.08** (−3.94)
- Variante warmup 12 pts): `results/sixeyes_periodo_adaptativo_warmup12.json`: **39.12** (−0.90)

## Autopsia
- v1: las pérdidas estaban en series de 5 min (realTweets −3.78), no en
  horarias (AdExchange +0.33). Causa: 200 puntos de fallback global sin
  estacionalidad perturban scores/ECDF/umbrales en todas las series.
- v2: las horarias EMPEORAN con 24 buckets (AdExchange −1.46) y realTweets
  mejora +6.71 con buckets idénticos (96). Patrón = ruido amplificado por el
  optimizador de umbrales, no señal.
- Falsación: los buckets vacíos son inocuos (nunca se consultan); colapsar
  96→24 no cambia la secuencia muestral de los buckets activos. La hipótesis
  estructural era falsa.

## Lección
No todo "defecto estético" (buckets muertos) es un defecto funcional. El
costo de cualquier warmup no estacional supera el beneficio cuando 48/58
series no necesitan cambio. La estacionalidad se aborda en la auditoría de latencia
(Holt-Winters) o no se ataca.

## Artefactos
- `src/sixeyes/experimental/adaptive_seasonal.py` (conservado como referencia)
- `tests/test_adaptive_seasonal.py` (7 passed, cubre causalidad/reset)
- Esta vía queda archivada en experimental/.
