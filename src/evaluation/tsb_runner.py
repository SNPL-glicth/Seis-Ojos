import os
import sys
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def get_tuning_files(list_file: str) -> List[str]:
    df = pd.read_csv(list_file)
    return df[~df['file_name'].str.contains('_NAB_')]['file_name'].tolist()

def _get_tsb_metrics(scores: np.ndarray, labels: np.ndarray, sliding_window: int) -> Dict[str, float]:
    tsb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'benchmarks', 'tsb_ad', 'TSB-AD'))
    if tsb_path not in sys.path:
        sys.path.append(tsb_path)
    
    from TSB_AD.evaluation.metrics import get_metrics
    from sklearn.preprocessing import MinMaxScaler
    
    if len(scores) > 0:
        scaled_scores = MinMaxScaler(feature_range=(0,1)).fit_transform(scores.reshape(-1, 1)).ravel()
    else:
        scaled_scores = scores
        
    pred = scaled_scores > (np.mean(scaled_scores) + 3*np.std(scaled_scores))
    return get_metrics(scaled_scores, labels, slidingWindow=sliding_window, pred=pred)

def run_single_series(detector_factory, file_path: str, timeout_s: float | None = 120.0) -> Dict[str, Any]:
    start_time = time.time()
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path).dropna()
        if df.empty:
            raise ValueError("Empty dataframe after dropna")
            
        data = df.iloc[:, 0:-1].values.astype(float)
        labels = df['Label'].astype(int).to_numpy()
        
        tsb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'benchmarks', 'tsb_ad', 'TSB-AD'))
        if tsb_path not in sys.path:
            sys.path.append(tsb_path)
            
        from TSB_AD.utils.slidingWindows import find_length_rank
        n_dim = data.shape[1]
        if n_dim == 1:
            sliding_window = find_length_rank(data, rank=1)
        else:
            sliding_window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
            
        detector = detector_factory()
        scores = []
        t_det0 = time.time()
        for i in range(len(data)):
            if timeout_s is not None and time.time() - start_time > timeout_s:
                return {"status": "NO_EVALUADA", "error": "Timeout", "time": time.time() - start_time, "length": 0, "metrics": {}, "time_detector": 0.0, "time_metrics": 0.0}
            
            val = float(data[i, 0])
            score = detector.update(val, str(i))
            if math.isnan(score) or math.isinf(score):
                return {"status": "NO_EVALUADA", "error": "Score is NaN/Inf", "time": time.time() - start_time, "length": 0, "metrics": {}, "time_detector": 0.0, "time_metrics": 0.0}
            scores.append(score)
        time_detector = time.time() - t_det0
            
        scores = np.array(scores)
        t_met0 = time.time()
        metrics = _get_tsb_metrics(scores, labels, sliding_window)
        time_metrics = time.time() - t_met0
        
        return {
            "status": "OK",
            "length": len(data),
            "metrics": metrics,
            "time": time.time() - start_time,
            "time_detector": time_detector,
            "time_metrics": time_metrics,
            "error": None
        }
    except Exception as e:
        return {"status": "NO_EVALUADA", "error": type(e).__name__ + ": " + str(e), "time": time.time() - start_time, "length": 0, "metrics": {}, "time_detector": 0.0, "time_metrics": 0.0}