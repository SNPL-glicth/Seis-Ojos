# Aprendizaje — registro vivo de experimentos Six Eyes

> Todo lo intentado, medido y archivado. Cada entrada: hipótesis → medición → autopsia → lección.
> Regla: ningún número sin artefacto en `results/`; ninguna vigencia sin medición superada.

## 1. Estado heredado (honesto)

- `results/sixeyes_tsb_fresh472.json` y `sixeyes_tsb_fresh472_rows.csv`: **INVÁLIDOS**. Nada de ahí se cita.
  Cifras anuladas: cand 0.713 (211 series), cand 0.507 (472), z-score 0.268, random 0.011.
- Motivo: AUC de random 0.011 en el lote (imposible: scores constantes dan 0.5 exacto) frente a 0.5087 en el chequeo individual.
- **La ubicación exacta del bug NO está probada.** "Escritura del CSV" es hipótesis no verificada. No se afirma que el detector esté libre de bug.
- **No hay scores guardados** (el CSV solo guardó arm/file/status/AUC-ROC/AUC-PR). Por tanto el plan es **re-ejecución con guardado de scores**, no "re-evaluación desde scores guardados".
- **Veredicto del candidato ECDF+bypass en TSB fresco: SIN VEREDICTO**. Los números están invalidados por bug de ruta por lotes.

## 2. Congelado (no se toca)

- Candidato: ECDF16000 + bypass 5%/100, sin PeakDecay ni refractario, semilla 42 (`cand_s42`).
- Conjunto: las 472 series frescas (disyunción con Eva y Tuning ya verificada).
- **Criterio de éxito (sin cambios):**
  - (a) diferencia media AUC-ROC cand − z-score con IC 95% bootstrap por serie que NO cruce 0, **y**
  - (b) >= 55% de series con AUC-ROC > 0.5.
  - Se evalúa sobre el conjunto pareado (series válidas en los tres brazos).
- **Métricas:** AUC-ROC y AUC-PR únicamente (sin VUS), con `sklearn.metrics`.

## 3. Plan en dos etapas

### Etapa 1 — generar y guardar scores

Para cada brazo (`cand_s42`, `zscore`, `random seed 42`) y cada serie:
```
np.savez_compressed(f"scores/{arm}/{name}.npz", s=scores, y=labels)
```
La limpieza de filas se aplica a scores y etiquetas a la vez.
- **Z-score:** medias/desviaciones rodantes vectorizadas (cumsum), misma definición que `ZScore_Raw`.

### Etapa 2 — evaluar desde disco, con UNA función para los tres brazos

```python
assert len(s)==len(y) and np.isfinite(s).all()
auc = roc_auc_score(y, s)
```
Si una serie falla una aserción o tiene una sola clase: fila con `status` y motivo, y se excluye para TODOS los brazos.

## 4. Puertas de sanidad (se evalúan ANTES de mirar al candidato)

- **G1 Random:** media AUC-ROC en [0.45, 0.55].
- **G2 Z-score:** media en [0.52, 0.66] (referencias: Tuning 0.598, Eva 0.584 con series distintas).
- **G3 Implementación del z-score:** en 5 series de Tuning, |AUC nuevo − AUC en `ZScore_Raw_tsb_results.csv`| < 0.01.
- **G4 Cobertura:** se reportan series válidas por brazo y motivos de exclusión; el conjunto pareado debe ser >= 80% de las 472.
- Si falla cualquier puerta: resultado INVÁLIDO, no se interpreta el candidato y se busca el bug.

## 5. Reglas

1. **Detector y criterio congelados.** Solo cambia la evaluación.
2. Esto es una **re-evaluación, no un segundo tiro limpio**: las 472 series ya se vieron una vez.
3. **Las 7 series de 50K–195K:** con `roc_auc_score` no deberían fallar; imprimir el traceback de una antes de excluirlas.
4. **Medir el tiempo de UNA serie antes de lanzar el lote.** El candidato cuesta ~0.11 ms/punto; el cuello previo fue el z-score y las métricas.
5. Si el resultado pasa o falla, se reporta tal cual, con el motivo de cada exclusión.

## 6. Predicción (escrita antes de medir)

Mi estimación de que se cumplan (a) y (b): **~40%**. Si falla, se archiva sin reajustes.

## 7. Resultado heredado (solo histórico, sin validez)

| Config | Series | AUC-ROC media | 55% series > 0.5 | Criterio |
|--------|--------|---------------|-----------------|----------|
| cand_s42 (211) | 211 | 0.713 | 90.05% | *datos sesgados, sin validar* |
| cand_s42 (472) | 472 | 0.507 | 53.3% | *invalidado por bug lote* |
| Full (472) | 472 | — | — | *sin evaluar* |

**Veredicto:** **SIN VEREDICTO**. El candidato ECDF+bypass no puede ser aprobado ni reprobado hasta completar la re-evaluación en dos etapas. El bug de ruta por lotes invalida los números anteriores.

## 8. Siguiente paso

Ejecutar Etapa 1 (generar scores) y Etapa 2 (evaluar desde disco). Una vez completado, aplicar puertas de sanidad (G1–G4) y reportar veredicto final. Sin reajustes, sin cambiar el detector ni el criterio.

## 9. Diagnóstico técnico (sin validar)

**Causa sin localizar; candidatas: construcción del brazo random, alineación scores-etiquetas, función de AUC del lote. Solo el random en 0.010 es imposible por sí mismo.**

**posible causa común; solo confirmado imposible el random.**

Z-score (0.358) y candidato (0.405) también bajo 0.5 por la misma ruta: la causa afecta a los tres brazos comúnmente, no solo al random.

**INVÁLIDO por puertas** (G1: random AUC media 0.010; G2: z-score AUC media 0.358; G4: 306/472). No existen scores .npz en disco. Las 306 series proceden de la ruta por lotes anterior. G3 pendiente de ejecución. Random 0.010 en 306/306 series es incompatible con scores independientes de las etiquetas: la causa está en la construcción o evaluación, sin localizar.

**sixeyes_tsb_fresh472.json: INVÁLIDO** — números sin validar por bug de ruta por lotes.

**Fin del acta técnica pending.**