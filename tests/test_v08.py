import json
from pathlib import Path
from aqx.risk_selector import RuntimeRiskModel,BACKENDS

def test_risk_model_predicts_valid_backend():
    cases=json.loads((Path(__file__).resolve().parents[1]/'results/source_v05_cases.json').read_text())
    model=RuntimeRiskModel(n_estimators=30,random_state=7,min_samples_leaf=1).fit(cases[:8])
    b,preds=model.predict(cases[8],lambda_risk=1.0)
    assert b in BACKENDS
    assert len(preds)==4
    assert all(x.predicted_time_s>0 for x in preds)
