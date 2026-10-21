from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
import gc
from typing import Any

import numpy as np

from .backends.multi import BACKENDS, get_backend, prepare_inputs
from .workload import WorkloadSpec


@dataclass
class BackendMeasurement:
    workload: str
    size: int
    backend: str
    repeat: int
    setup_s: float
    execute_s: float
    total_s: float
    value: float
    reference_error: float | None = None

    def to_dict(self):
        return asdict(self)


def warmup_backend(workload: str, backend: str, size: int) -> float:
    args, kwargs = prepare_inputs(workload, min(size, 64))
    return float(get_backend(workload, backend)(*args, **kwargs))


def measure(workload: str, backend: str, size: int, repeats: int = 3, warmup: int = 1, seed: int = 7) -> list[BackendMeasurement]:
    args, kwargs = prepare_inputs(workload, size, seed=seed)
    if warmup:
        for _ in range(warmup):
            get_backend(workload, backend)(*args, **kwargs)
    rows = []
    for repeat in range(repeats):
        gc.collect()
        t0 = perf_counter()
        value = float(get_backend(workload, backend)(*args, **kwargs))
        execute = perf_counter() - t0
        rows.append(BackendMeasurement(workload, size, backend, repeat, 0.0, execute, execute, value))
    return rows


def evaluate_case(workload: str, size: int, repeats: int = 3, seed: int = 7) -> tuple[list[BackendMeasurement], WorkloadSpec]:
    spec = WorkloadSpec(**__import__('aqx.workload', fromlist=['make_spec']).make_spec(workload, size).to_dict())
    all_rows: list[BackendMeasurement] = []
    reference = None
    for backend in BACKENDS:
        rows = measure(workload, backend, size, repeats=repeats, seed=seed)
        median = float(np.median([r.execute_s for r in rows]))
        if reference is None:
            reference = rows[0].value
        for r in rows:
            r.reference_error = abs(r.value - reference)
        all_rows.extend(rows)
    return all_rows, spec
