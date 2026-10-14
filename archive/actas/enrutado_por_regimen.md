# Router dt≥45min→predictor — SIN MEJORA (2026-10-14, archivado con honor)

## Resultado (ejecución única, reglas congeladas pre-código)
NAB: **45.13 (−0.06)** | Low FP 32.91 | Low FN 51.61 (+0.79).
AdExchange +3.41 (el router acertó donde apuntaba); KnownCause −2.40,
Tweets −0.81 por re-optimización del umbral global (0.99769 vs 0.99781).

## Lección
El mecanismo funciona; el benchmark lo absorbe. Umbral global único ⇒ las
redistribuciones locales son invariantes en el total. Umbrales por régimen
= tunear el benchmark. Fin de la vía de enrutado.

## Artefactos
- `experimental/regime_router.py`, `tests/test_regime_router.py` (6)
- `archive/gate_enrutado_regimen.py`, `results/sixeyes_enrutado_regimen.json`
- Configuración oficial vigente (dualscale, 45.19).
