# BASELINE vigente (configuración oficial, 2026-10-14)

Artefacto oficial inmutable:
- `results/sixeyes_dualscale_b15_k12_tau2p0_w16000.json`
- standard **45.19** | Low FN **50.82** | Low FP **34.16**
- Comando: `python run_nab.py --nab-path ../NAB` (pipeline
  `SeasonalDualScaleRobustZ + PercentileCalibrator + StreamingPeakDecay`,
  K=12, tau=2.0, cal W=16000; determinista bit a bit en re-ejecución)
- Medido: 2026-10-14. `final_results.json` oficial verificado sin mutación.

Historial de baselines (artefactos conservados):
- `results/sixeyes_peak_decay_b15_k12_tau2p0_w16000.json`: 40.02
  (`--detector seasonal_decay`). Diferencia: +5.17 pts.
- K=160 in-sample `results/sixeyes_peak_decay_b15_k160_tau1p5_wInf.json`:
  42.54 (`--cal-window 0`, coste O(N), sin claim de generalización).

Reglas (carta `opencode.md` §2–§5):
1. Ningún experimento modifica `src/sixeyes/detectors/`; todo nace en `src/sixeyes/experimental/`.
2. Promoción solo con gate numérico superado en LOCO-CV + multiseedilla.
3. Todo número citado requiere artefacto aislado en `results/`; se prohíbe tunear sobre las 58 series.
