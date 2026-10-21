from __future__ import annotations

import json
from pathlib import Path
import csv

from aqx.backends.multi import BACKENDS
from aqx.evaluator import evaluate_case
from aqx.selection import run_selector, add_selection_fields

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

WORKLOADS = {
    "monte_carlo": [256, 1024, 4096, 16384],
    "black_scholes": [256, 1024, 4096, 16384],
    "rolling_signal": [512, 2048, 8192, 32768],
    "portfolio_risk": [256, 1024, 4096, 16384],
}


def main():
    case_rows = []
    raw_rows = []
    for workload, sizes in WORKLOADS.items():
        for size in sizes:
            measurements, spec = evaluate_case(workload, size, repeats=3)
            medians = {b: min(1e9, __import__('numpy').median([m.execute_s for m in measurements if m.backend == b])) for b in BACKENDS}
            row = {"workload": workload, "size": size, "spec": spec.to_dict(), "medians": medians}
            case_rows.append(row)
            for m in measurements:
                raw_rows.append(m.to_dict())

    # Train on the first 60% of each workload so held-out cases include the same families.
    train, test = [], []
    for workload in WORKLOADS:
        family = [r for r in case_rows if r["workload"] == workload]
        cut = max(1, int(len(family) * 0.5))
        train.extend(family[:cut])
        test.extend(family[cut:])
    import numpy as np
    from aqx.cost_model import AdaptiveCostModel
    model = AdaptiveCostModel(alpha=0.25)
    model.fit(train)
    selected = []
    for case in test:
        pred = model.predict(__import__('aqx.workload', fromlist=['WorkloadSpec']).WorkloadSpec(**case["spec"]))
        backend = model.select(__import__('aqx.workload', fromlist=['WorkloadSpec']).WorkloadSpec(**case["spec"]))
        selected.append(add_selection_fields(case, backend, case["medians"], pred))

    summary = {
        "version": "0.2",
        "backends": list(BACKENDS),
        "workloads": {k: v for k, v in WORKLOADS.items()},
        "n_cases": len(case_rows),
        "n_test": len(selected),
        "selection_accuracy": float(np.mean([r["selection_correct"] for r in selected])),
        "mean_regret": float(np.mean([r["regret"] for r in selected])),
        "median_regret": float(np.median([r["regret"] for r in selected])),
        "max_regret": float(np.max([r["regret"] for r in selected])),
        "mean_oracle_speedup_vs_python": float(np.mean([
            r["medians"]["python"] / r["oracle_time_s"] for r in selected
        ])),
    }

    with open(RESULTS / "v02_cases.json", "w") as f:
        json.dump(case_rows, f, indent=2)
    with open(RESULTS / "v02_test_selection.json", "w") as f:
        json.dump(selected, f, indent=2)
    with open(RESULTS / "v02_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS / "v02_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
        writer.writeheader(); writer.writerows(raw_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
