import json
import math
import bisect
from collections import deque
import numpy as np

class MockInternalDetector:
    def update(self, val, ts):
        return val

class RawPercentileCalibrator:
    def __init__(self, max_history=300, min_historial=50):
        self._historial = []
        self._cola = deque()
        self._max_history = max_history
        self._min_historial = min_historial

    def _evict_if_full(self):
        if self._max_history is None:
            return
        if len(self._historial) >= self._max_history and self._cola:
            oldest = self._cola.popleft()
            idx = bisect.bisect_left(self._historial, oldest)
            for j in range(idx, min(idx + 4, len(self._historial))):
                if self._historial[j] == oldest:
                    self._historial.pop(j)
                    break
            else:
                try:
                    self._historial.remove(oldest)
                except ValueError:
                    pass

    def update(self, raw):
        if math.isnan(raw) or math.isinf(raw) or raw == 0.0:
            return 0.0
            
        if len(self._historial) < self._min_historial:
            bisect.insort(self._historial, raw)
            self._cola.append(raw)
            return 0.0
            
        menores = bisect.bisect_left(self._historial, raw)
        iguales = bisect.bisect_right(self._historial, raw) - menores
        n = len(self._historial)
        
        score = (menores + 0.5 * iguales + 0.5) / (n + 1)
        self._evict_if_full()
        bisect.insort(self._historial, raw)
        self._cola.append(raw)
        return score

def main():
    np.random.seed(42)
    calibrator = RawPercentileCalibrator(max_history=300, min_historial=50)
    
    # Generar 1,200 observaciones (900 pasos bajo expulsión continua)
    steps = []
    # Generamos una serie no trivial con ruido, saltos y empates exactos
    values = []
    for t in range(1200):
        val = 10.0 + 5.0 * math.sin(t * 0.05) + (t % 13) * 0.25 + np.random.normal(0, 0.5)
        if t % 50 == 0:
            val = round(val, 2) # inducir empates exactos
        values.append(val)
        
    for t, val in enumerate(values):
        score = calibrator.update(val)
        steps.append({
            "t": t,
            "raw": val,
            "score": score,
            "hist_len": len(calibrator._historial)
        })
        
    out_path = "RELIC/crates/relic_core/src/anomaly/data/golden_ecdf_eviction_1200.json"
    with open(out_path, "w") as f:
        json.dump(steps, f, indent=2)
    print(f"Exportados {len(steps)} pasos a {out_path} (capacidad=300, min=50).")

if __name__ == "__main__":
    main()

