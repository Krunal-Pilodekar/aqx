import json
from pathlib import Path

def test_v07_summary_exists_and_is_bounded():
    p=Path(__file__).resolve().parents[1]/'results'/'v07_summary.json'
    d=json.loads(p.read_text())
    assert d['n_cases']==16
    assert d['n_model_predictions']==80
    assert 0.0 <= d['rf']['accuracy_mean'] <= 1.0
    assert d['rf']['mean_regret'] >= 0
    assert d['rf']['max_regret'] >= d['rf']['mean_regret']

def test_v07_prediction_count():
    p=Path(__file__).resolve().parents[1]/'results'/'v07_predictions.json'
    rows=json.loads(p.read_text())
    assert len(rows)==80
