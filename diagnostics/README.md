# Herramientas de Diagnóstico e Inspección (`diagnostics/`)

Este directorio contiene herramientas de diagnóstico estadístico, trazabilidad de señales, visualización interactiva y verificación de invariantes desarrolladas para auditar el comportamiento interno de los estimadores de **Six Eyes**.

A diferencia de los scripts de `scratch/` (centrados en hipótesis de benchmark y optimización de puntajes), las utilidades de este módulo inspeccionan la física de las señales, la estabilidad numérica de los estimadores robustos (MAD/mediana) y la calidad de las distribuciones de scores generadas.

---

## Índice de Herramientas y Propósito

### 1. Inspección de Estabilidad Numérica y Estimadores Robustos

* **`diagnostic_2.py`:**
  * **Propósito:** Diagnóstico del comportamiento del estimador no paramétrico en streaming (`DiagnosticRobustZ`).
  * **Métricas evaluadas:** Seguimiento continuo de la mediana móvil, cálculo de la Desviación Absoluta respecto a la Mediana (MAD), detección de colapsos de escala local ($\text{MAD} = 0$) y clasificación de estados del detector (`warmup`, `nan`, normal o outlier).
* **`diagnostic_v2.py`:**
  * **Propósito:** Auditoría avanzada de dispersión usando desviación media absoluta (`Mean Absolute Deviation`) y resolución observada (`DiagnosticRobustZ_v2`).
  * **Mecanismo:** Evalúa la cuantización mínima de la señal para evitar singularidades por división espuria en señales con escalonamiento discreto.
* **`diagnostic_4.py`:**
  * **Propósito:** Auditoría interna de canastas diurnas en `DailySeasonalRobustZ`.
  * **Métricas evaluadas:** Comprueba el calentamiento de cada canasta de 15 minutos (96 casillas diarias), mide la frecuencia de colapso de escala por hora del día y audita la sensibilidad local en ventanas de fallo industrial.

---

### 2. Calidad de Distribuciones y Cobertura de Verdad Fundamental

* **`diagnostic.py`:**
  * **Propósito:** Análisis de distribución, densidad y cardinalidad de los scores de anomalía emitidos.
  * **Métricas evaluadas:** Proporción de ceros exactos emitidos, cuantiles de saturación ($0.5$, $0.9$, $0.99$, $1.0$), presencia de valores `NaN`/`Inf` y resolución decimal de las salidas para garantizar ranking continuo.
* **`diagnostic_3.py`:**
  * **Propósito:** Perfilado de granularidad temporal y densidad muestral en las 58 series de NAB.
  * **Métricas evaluadas:** Mapeo de frecuencias de muestreo (1 min, 5 min, 1 h), conteo total de puntos y distribución de ventanas de verdad fundamental por categoría.
* **`diagnostic_5.py`:**
  * **Propósito:** Comparación de cobertura de ventanas entre variantes de detectores (`zscore_v3` vs `seasonal`) bajo umbrales extremadamente conservadores ($T = 0.9999$).
  * **Mecanismo:** Rastrea exactamente cuáles ventanas de anomalía logran ser alcanzadas y cuáles quedan silenciadas por sobre-atenuación.

---

### 3. Verificación de Causalidad y Trazabilidad de Señales

* **`diagnostic_6.py`:**
  * **Propósito:** Verificación de no-fuga causal en `DailySeasonalRobustZ`.
  * **Metodología:** Inyecta un choque extremo artificial en $t \ge k$ y comprueba bit a bit que las salidas para el intervalo pasado $t < k$ permanezcan estrictamente idénticas al flujo original.

---

### 4. Generación y Visualización Interactiva

* **`generate_html.py`:**
  * **Propósito:** Generador de reportes visuales interactivos independientes en HTML utilizando `Chart.js`.
  * **Mecanismo:** Cruza la serie temporal original, los scores de anomalía emitidos por el modelo y las ventanas de verdad fundamental de NAB, generando gráficos interactivos con zoom y tooltips para inspección visual de falsos positivos y retardos de detección.
* **`diagnostic_charts.html`:**
  * **Propósito:** Dashboard visual autocontenido resultante de la ejecución de `generate_html.py`, utilizado para auditar visualmente casos complejos de telemetría (como demanda de taxis en NYC o fallos en máquinas industriales).
