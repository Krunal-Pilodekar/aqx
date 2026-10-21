import json
from pathlib import Path

from aqx.calibration import fit_scaling


def test_fit_scaling_reconstructs_linear_power_law():
    pred = fit_scaling([(10, 1.0), (100, 10.0)], 1000)
    assert 90 < pred.predicted_execute_s < 110
    assert abs(pred.slope - 1.0) < 1e-9


def test_v03_summary_contract():
    p = Path(__file__).resolve().parents[1] / "results" / "v03_summary.json"
    if p.exists():
        data = json.loads(p.read_text())
        for key in ["online_selection_accuracy", "online_mean_regret", "mean_calibration_s"]:
            assert key in data
