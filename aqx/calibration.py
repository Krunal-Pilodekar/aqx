from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
import gc
import math
from typing import Any

import numpy as np

from .backends.multi import BACKENDS, get_backend, prepare_inputs


@dataclass
class PilotMeasurement:
    workload: str
    target_size: int
    pilot_size: int
    backend: str
    repeat: int
    execute_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationPrediction:
    backend: str
    predicted_execute_s: float
    slope: float
    intercept: float
    pilot_points: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pilot_sizes(target_size: int, points: int = 2) -> list[int]:
    if target_size <= 1:
        return [1]
    fractions = [0.125, 0.25, 0.5][: max(1, points)]
    sizes = sorted({max(16, min(target_size, int(round(target_size * f)))) for f in fractions})
    if sizes[-1] == target_size and len(sizes) > 1:
        sizes = sizes[:-1]
    return sorted(set(sizes)) or [target_size]


def measure_pilot(
    workload: str,
    backend: str,
    target_size: int,
    pilot_size: int,
    repeats: int = 2,
    seed: int = 7,
) -> list[PilotMeasurement]:
    args, kwargs = prepare_inputs(workload, pilot_size, seed=seed)
    fn = get_backend(workload, backend)
    # Warm once at the pilot size so that the measured pilot represents steady-state execution.
    fn(*args, **kwargs)
    rows: list[PilotMeasurement] = []
    for repeat in range(repeats):
        gc.collect()
        t0 = perf_counter()
        fn(*args, **kwargs)
        rows.append(PilotMeasurement(workload, target_size, pilot_size, backend, repeat, perf_counter() - t0))
    return rows


def fit_scaling(points: list[tuple[int, float]], target_size: int) -> CalibrationPrediction:
    if not points:
        raise ValueError("No calibration points")
    clean = [(float(s), max(float(t), 1e-12)) for s, t in points]
    if len(clean) == 1:
        s, t = clean[0]
        slope = 1.0
        intercept = math.log(t) - slope * math.log(s)
    else:
        x = np.log(np.array([p[0] for p in clean], dtype=float))
        y = np.log(np.array([p[1] for p in clean], dtype=float))
        A = np.column_stack([np.ones(len(x)), x])
        intercept, slope = np.linalg.lstsq(A, y, rcond=None)[0]
    predicted = float(np.exp(intercept + slope * math.log(max(target_size, 1))))
    return CalibrationPrediction("", predicted, float(slope), float(intercept), [{"size": s, "median_s": t} for s, t in clean])


def calibrate_backend(
    workload: str,
    backend: str,
    target_size: int,
    pilot_points: int = 2,
    repeats: int = 2,
    seed: int = 7,
) -> tuple[CalibrationPrediction, list[PilotMeasurement], float]:
    start = perf_counter()
    measurements: list[PilotMeasurement] = []
    for pilot_size in _pilot_sizes(target_size, pilot_points):
        measurements.extend(measure_pilot(workload, backend, target_size, pilot_size, repeats, seed))
    points: list[tuple[int, float]] = []
    for pilot_size in sorted({m.pilot_size for m in measurements}):
        vals = [m.execute_s for m in measurements if m.pilot_size == pilot_size]
        points.append((pilot_size, float(np.median(vals))))
    pred = fit_scaling(points, target_size)
    pred.backend = backend
    return pred, measurements, perf_counter() - start


class OnlineCalibrator:
    """Runtime calibrator that uses cheap pilot executions to estimate full-size cost."""

    def __init__(self, candidate_backends: tuple[str, ...] = BACKENDS, pilot_points: int = 2, repeats: int = 2):
        self.candidate_backends = tuple(candidate_backends)
        self.pilot_points = int(pilot_points)
        self.repeats = int(repeats)

    def select(self, workload: str, target_size: int, candidates: tuple[str, ...] | None = None, seed: int = 7):
        candidates = tuple(candidates or self.candidate_backends)
        predictions: dict[str, CalibrationPrediction] = {}
        measurements: list[PilotMeasurement] = []
        overhead = 0.0
        for backend in candidates:
            pred, rows, elapsed = calibrate_backend(
                workload, backend, target_size,
                pilot_points=self.pilot_points, repeats=self.repeats, seed=seed,
            )
            predictions[backend] = pred
            measurements.extend(rows)
            overhead += elapsed
        selected = min(predictions, key=lambda b: predictions[b].predicted_execute_s)
        return selected, predictions, measurements, overhead
