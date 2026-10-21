from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqx.backends.multi import BACKENDS
from aqx.cost_model import AdaptiveCostModel
from aqx.evaluator import evaluate_case
from aqx.online_selector import AdaptiveOnlineSelector

WORKLOADS = {
    "monte_carlo": [512, 2048, 8192, 16384],
    "black_scholes": [512, 2048, 8192, 16384],
    "rolling_signal": [1024, 4096, 16384, 32768],
    "portfolio_risk": [512, 2048, 8192, 16384],
}

RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)


def build_cases():
    cases, raw = [], []
    for workload, sizes in WORKLOADS.items():
        for size in sizes:
            measurements, spec = evaluate_case(workload, size, repeats=3)
            medians = {b: float(np.median([m.execute_s for m in measurements if m.backend == b])) for b in BACKENDS}
            oracle = min(medians, key=medians.get)
            cases.append({"workload": workload, "size": size, "spec": spec.to_dict(), "medians": medians, "oracle": oracle})
            raw.extend(m.to_dict() for m in measurements)
    return cases, raw


def main():
    cases, raw = build_cases()
    train, test = [], []
    for workload in WORKLOADS:
        fam = [c for c in cases if c["workload"] == workload]
        train.extend(fam[:2])
        test.extend(fam[2:])

    model = AdaptiveCostModel(alpha=0.25)
    model.fit(train)
    selector = AdaptiveOnlineSelector(model, top_k=3, pilot_points=2, repeats=2)

    rows = []
    for case in test:
        from aqx.workload import WorkloadSpec
        spec = WorkloadSpec(**case["spec"])
        static_pred = model.predict(spec)
        static_selected = model.select(spec)
        online = selector.select(spec)
        oracle_t = case["medians"][case["oracle"]]
        selected_t = case["medians"][online.selected]
        static_t = case["medians"][static_selected]
        row = {
            "workload": case["workload"], "size": case["size"], "oracle": case["oracle"],
            "static_selected": static_selected, "online_selected": online.selected,
            "static_correct": static_selected == case["oracle"],
            "online_correct": online.selected == case["oracle"],
            "static_regret": static_t / oracle_t - 1.0,
            "online_regret": selected_t / oracle_t - 1.0,
            "online_selected_time_s": selected_t,
            "oracle_time_s": oracle_t,
            "calibration_s": online.calibration_s,
            "predicted_times": static_pred,
            "calibrated_predicted_times": online.calibrated_predictions,
            "candidates": list(online.candidates),
        }
        rows.append(row)

    summary = {
        "version": "0.3",
        "n_total_cases": len(cases),
        "n_test": len(test),
        "top_k": selector.top_k,
        "pilot_points": selector.calibrator.pilot_points,
        "pilot_repeats": selector.calibrator.repeats,
        "static_selection_accuracy": float(np.mean([r["static_correct"] for r in rows])),
        "online_selection_accuracy": float(np.mean([r["online_correct"] for r in rows])),
        "static_mean_regret": float(np.mean([r["static_regret"] for r in rows])),
        "online_mean_regret": float(np.mean([r["online_regret"] for r in rows])),
        "static_median_regret": float(np.median([r["static_regret"] for r in rows])),
        "online_median_regret": float(np.median([r["online_regret"] for r in rows])),
        "online_max_regret": float(np.max([r["online_regret"] for r in rows])),
        "mean_calibration_s": float(np.mean([r["calibration_s"] for r in rows])),
        "median_calibration_s": float(np.median([r["calibration_s"] for r in rows])),
        "fraction_regret_improved": float(np.mean([r["online_regret"] < r["static_regret"] for r in rows])),
    }

    (RESULTS / "v03_cases.json").write_text(json.dumps(cases, indent=2))
    (RESULTS / "v03_online_selection.json").write_text(json.dumps(rows, indent=2))
    (RESULTS / "v03_summary.json").write_text(json.dumps(summary, indent=2))
    with (RESULTS / "v03_raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw[0].keys())
        writer.writeheader(); writer.writerows(raw)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
