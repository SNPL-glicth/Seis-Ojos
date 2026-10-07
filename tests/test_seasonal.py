from detectors.seasonal import DailySeasonalRobustZ

def test_daily_seasonal_robusto():
    detector = DailySeasonalRobustZ(bucket_size_minutes=15, window_size=5, min_observations=2)
    
    # 07:15 = 7*60+15 = 435. 435/15 = 29 (bucket index)
    score = detector.update(10.0, "2014-04-10 07:15:00")
    assert score == 0.0  # Warmup
    score = detector.update(12.0, "2014-04-11 07:15:00")
    assert score == 0.0  # Warmup complete for bucket 29 (2 obs)
    
    # 08:30 = 8*60+30 = 510. 510/15 = 34
    score = detector.update(20.0, "2014-04-11 08:30:00")
    assert score == 0.0  # Warmup for bucket 34
    
    # Back to bucket 29, it should have enough observations now.
    # median = 11.0, mad = 1.0
    score = detector.update(30.0, "2014-04-12 07:15:00")
    assert score > 0.0  # Anomaly compared to [10, 12]

def test_asignacion_bucket():
    detector = DailySeasonalRobustZ(bucket_size_minutes=60) # 1 hour buckets
    # 00:00 to 00:59 should be bucket 0
    assert detector._parse_and_bucket("2020-01-01 00:00:00") == 0
    assert detector._parse_and_bucket("2020-01-01 00:59:59") == 0
    # 01:00 to 01:59 should be bucket 1
    assert detector._parse_and_bucket("2020-01-01 01:00:00") == 1
    
    # Check fallback format support just in case
    assert detector._parse_and_bucket("2020-01-01 23:59:59.123") == 23

def test_instancias_independientes():
    detector = DailySeasonalRobustZ(bucket_size_minutes=15)
    detector.update(10.0, "2020-01-01 08:00:00")
    detector.update(10.0, "2020-01-01 09:00:00")
    
    b1 = detector._parse_and_bucket("2020-01-01 08:00:00")
    b2 = detector._parse_and_bucket("2020-01-01 09:00:00")
    
    assert detector._buckets[b1] is not detector._buckets[b2]
    
def test_anti_fuga_seasonal():
    # Verifica que procesar un valor futuro (ej en otro bucket, o mas tarde) 
    # no cambia el calculo del pasado
    det_a = DailySeasonalRobustZ(bucket_size_minutes=60, window_size=5, min_observations=2)
    det_b = DailySeasonalRobustZ(bucket_size_minutes=60, window_size=5, min_observations=2)
    
    # Serie normal para A
    s_a = []
    s_a.append(det_a.update(10.0, "2020-01-01 08:00:00"))
    s_a.append(det_a.update(12.0, "2020-01-02 08:00:00"))
    s_a.append(det_a.update(30.0, "2020-01-03 08:00:00"))
    
    # Serie para B con fuga ficticia en t2
    s_b = []
    s_b.append(det_b.update(10.0, "2020-01-01 08:00:00"))
    # Metemos algo en otro bucket para simular paso del tiempo
    det_b.update(100.0, "2020-01-01 09:00:00")
    
    s_b.append(det_b.update(12.0, "2020-01-02 08:00:00"))
    s_b.append(det_b.update(30.0, "2020-01-03 08:00:00"))
    
    assert s_a == s_b
