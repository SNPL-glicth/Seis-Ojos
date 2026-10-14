# P&L por archivo (auditoría offline, 2026-10-14)

Fuente: `sixeyes_dualscale_b15_k12_tau2p0_w16000_standard_scores.csv`
(raw scores por archivo; umbral global 0.99781).

## Totales por categoría (raw)
- artificialNoAnomaly −0.44 (5) · artificialWithAnomaly +0.94 (6)
- realAWSCloudwatch −1.17 (17) · realAdExchange −4.04 (6, peor media −0.67)
- realKnownCause −3.41 (7) · realTraffic −0.93 (7) · realTweets −2.11 (10)

## Top ladrones (TP/FP/FN)
| Archivo | Score | TP | FP | FN |
|---|---|---|---|---|
| realTweets/Twitter_volume_UPS.csv | −3.79 | 3 | 6 | 1582 |
| realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv | −3.11 | 0 | 1 | 405 |
| realAdExchange/exchange-3_cpc_results.csv | −3.00 | 0 | 0 | 153 |
| realTweets/Twitter_volume_IBM.csv | −2.89 | 5 | 44 | 1585 |
| realKnownCause/rogue_agent_key_hold.csv | −2.11 | 0 | 1 | 190 |
| realTraffic/speed_7578.csv | −2.11 | 1 | 0 | 115 |

## Hallazgo director
El drenaje es **FN (ventanas perdidas)**, no FPs: los 6 peores suman
~15 FPs contra ~4000 FNs. Hipótesis refutada: `artificialNoAnomaly` solo
−0.44 (el umbral 0.9978 + ECDF W16000 ya mantienen quietos los limpios).
La calibración en reposo (punto 2 del backlog) pierde prioridad.

## Implicación
El siguiente gate debe subir TPs: retraso de detección intra-ventana
(punto 3) y predictor + likelihood para patrones que la densidad no ve
(punto 6: `rogue_agent_key_hold/updown`, `exchange-3_cpc` con TP=0 y FP=0
son ceguera total, no ruido). Top aportantes (AAPL +2.86, AMZN +1.97,
ec2_request_latency +2.07) marcan el patrón que sí capturamos: spikes
aislados sobre línea estable.
