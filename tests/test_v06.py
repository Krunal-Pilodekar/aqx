import json
from pathlib import Path
import numpy as np
from aqx.ml_selector import BackendClassifier
from aqx.workload import make_spec

def test_feature_model_trains_on_synthetic_records():
    rows=[]
    for w in ['monte_carlo','black_scholes','rolling_signal','portfolio_risk']:
        for s in [256,512,1024,2048]:
            spec=make_spec(w,s).to_dict()
            med={'python':1.0,'numpy':0.5,'numba':0.2,'numba_parallel':0.3}
            rows.append({'spec':spec,'oracle':'numba','medians':med})
    m=BackendClassifier(n_estimators=30,random_state=1); m.fit(rows)
    p=m.predict(make_spec('monte_carlo',4096)); assert abs(sum(p.probabilities.values())-1)<1e-8

def test_prediction_is_deterministic():
    rows=[]
    for s in [256,512,1024,2048,4096,8192]:
        spec=make_spec('black_scholes',s).to_dict()
        med={'python':1.0,'numpy':0.7,'numba':0.2,'numba_parallel':0.3}
        rows.append({'spec':spec,'oracle':'numba','medians':med})
    m=BackendClassifier(n_estimators=40,random_state=3); m.fit(rows)
    p1=m.predict(make_spec('black_scholes',16384)); p2=m.predict(make_spec('black_scholes',16384))
    assert p1.selected==p2.selected and p1.probabilities==p2.probabilities
