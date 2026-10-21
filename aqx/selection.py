from __future__ import annotations

import numpy as np

from .backends.multi import BACKENDS
from .cost_model import AdaptiveCostModel
from .workload import WorkloadSpec


def summarize_case(measurements) -> dict:
    medians = {}
    for backend in BACKENDS:
        vals = [m.execute_s for m in measurements if m.backend == backend]
        medians[backend] = float(np.median(vals))
    oracle = min(medians, key=medians.get)
    return {"medians": medians, "oracle": oracle, "oracle_time_s": medians[oracle]}


def add_selection_fields(row: dict, selected: str, medians: dict[str, float], predictions: dict[str, float] | None = None) -> dict:
    oracle = min(medians, key=medians.get)
    oracle_t = medians[oracle]
    row = dict(row)
    row.update({
        "oracle": oracle,
        "selected": selected,
        "selection_correct": selected == oracle,
        "regret": medians[selected] / oracle_t - 1.0,
        "selected_time_s": medians[selected],
        "oracle_time_s": oracle_t,
    })
    if predictions is not None:
        row["predictions"] = predictions
    return row


def run_selector(cases: list[dict], training_fraction: float = 0.6) -> tuple[list[dict], AdaptiveCostModel]:
    n_train = max(1, int(len(cases) * training_fraction))
    train = cases[:n_train]
    test = cases[n_train:]
    model = AdaptiveCostModel(alpha=0.25)
    model.fit(train)
    out = []
    for case in test:
        spec = WorkloadSpec(**case["spec"])
        pred = model.predict(spec)
        selected = model.select(spec)
        out.append(add_selection_fields(case, selected, case["medians"], pred))
    return out, model
