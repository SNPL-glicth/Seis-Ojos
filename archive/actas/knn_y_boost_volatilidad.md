# FALLIDAS (archivadas 2026-10-14, archivado)

## 35.16 (−10.03)
Rescata UPS/KO/FB/fe7f93 pero hunde KnownCause −16.29. Sin normalización
por régimen, la heterogeneidad parece anomalía. Tercer componente
complementario que falla solo. `results/sixeyes_knn_retardos.json`.

## Muerta en gate barato
Retardo mediano 77→77.5 (proxy 0.99), cola IBM +42%. La volatilidad corta
es ubicua: el ratio no marca inicios de evento. Sin NAB completo.

## Lección conjunta
Dos flancos ortogonales, mismo veredicto: señal complementaria sin
operador de integración válido ni régimen que la contenga. El 45.19 sigue
vigente. Próximas vías: enrutado por régimen con reglas a priori,
k-NN con normalización, TSB-Eva como juez de generalización.

## Artefactos
- `experimental/knn_detector.py`, `experimental/volatility_boost.py`
- `tests/test_knn.py` (5), `tests/test_volatility_boost.py` (4)
- `archive/gate_knn.py`, `results/sixeyes_knn_retardos.json`
