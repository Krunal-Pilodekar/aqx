from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any
import gc
import math

import numpy as np

from .backends.multi import BACKENDS, get_backend, prepare_inputs


@dataclass
class Probe:
    backend: str
    size: int
    median_s: float
    repeats: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    backend: str
    predicted_s: float
    slope: float
    residual_log_rmse: float
    probes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe_sizes(target: int, fractions=(0.03125, 0.0625, 0.125, 0.25)) -> list[int]:
    if target <= 32:
        return [target]
    sizes = []
    for f in fractions:
        s = max(32, int(round(target * f)))
        s = min(target, s)
        sizes.append(s)
    return sorted(set(sizes))


def _median_time(workload: str, backend: str, size: int, repeats: int, seed: int) -> float:
    args, kwargs = prepare_inputs(workload, size, seed=seed)
    fn = get_backend(workload, backend)
    fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = perf_counter()
        fn(*args, **kwargs)
        times.append(perf_counter() - t0)
    return float(np.median(times))


def _fit(points: list[Probe], target: int) -> Prediction:
    if not points:
        raise ValueError("at least one probe is required")
    x = np.log(np.array([max(p.size, 1) for p in points], dtype=float))
    y = np.log(np.maximum(np.array([p.median_s for p in points], dtype=float), 1e-12))
    if len(points) == 1:
        slope = 1.0
        intercept = y[0] - slope * x[0]
        residual = 0.0
    else:
        A = np.column_stack([np.ones(len(x)), x])
        intercept, slope = np.linalg.lstsq(A, y, rcond=None)[0]
        residual = float(np.sqrt(np.mean((y - (intercept + slope * x)) ** 2)))
    pred = float(np.exp(intercept + slope * math.log(max(target, 1))))
    return Prediction("", pred, float(slope), residual, [p.to_dict() for p in points])


class MultiFidelityCalibrator:
    """Sequentially probes only promising/uncertain backends."""

    def __init__(
        self,
        candidates: tuple[str, ...] = BACKENDS,
        max_probes: int = 2,
        initial_fraction: float = 0.0625,
        uncertainty_ratio: float = 0.15,
        repeats: int = 1,
        seed: int = 7,
    ):
        self.candidates = tuple(candidates)
        self.max_probes = int(max_probes)
        self.initial_fraction = float(initial_fraction)
        self.uncertainty_ratio = float(uncertainty_ratio)
        self.repeats = int(repeats)
        self.seed = int(seed)

    def select(self, workload: str, target_size: int, ordered_candidates: tuple[str, ...] | None = None):
        candidates = tuple(ordered_candidates or self.candidates)
        start = perf_counter()
        probe_sizes = _probe_sizes(target_size)
        if target_size > 32:
            initial = max(32, min(target_size, int(round(target_size * self.initial_fraction))))
        else:
            initial = target_size
        if initial not in probe_sizes:
            probe_sizes = sorted(set([initial] + probe_sizes))

        probes: dict[str, list[Probe]] = {b: [] for b in candidates}
        predictions: dict[str, Prediction] = {}

        # Stage 1: one cheap probe for all shortlisted candidates.
        for b in candidates:
            t = _median_time(workload, b, initial, self.repeats, self.seed)
            probes[b].append(Probe(b, initial, t, self.repeats))
            p = _fit(probes[b], target_size); p.backend = b
            predictions[b] = p

        # Stage 2+: only spend extra probes when top predictions are too close or fits are uncertain.
        used_sizes = {b: initial for b in candidates}
        for next_size in probe_sizes:
            if next_size <= initial:
                continue
            ranked = sorted(candidates, key=lambda b: predictions[b].predicted_s)
            best = predictions[ranked[0]].predicted_s
            second = predictions[ranked[1]].predicted_s if len(ranked) > 1 else float("inf")
            close = second <= best * (1.0 + self.uncertainty_ratio)
            noisy = any(predictions[b].residual_log_rmse > 0.10 for b in ranked[: min(2, len(ranked))])
            if not (close or noisy):
                break
            chosen = ranked[: min(2, len(ranked))]
            for b in chosen:
                if len(probes[b]) >= self.max_probes:
                    continue
                t = _median_time(workload, b, next_size, self.repeats, self.seed)
                probes[b].append(Probe(b, next_size, t, self.repeats))
                p = _fit(probes[b], target_size); p.backend = b
                predictions[b] = p
            if all(len(probes[b]) >= self.max_probes for b in chosen):
                # Continue only if the top two are still genuinely ambiguous.
                ranked = sorted(candidates, key=lambda b: predictions[b].predicted_s)
                best = predictions[ranked[0]].predicted_s
                second = predictions[ranked[1]].predicted_s if len(ranked) > 1 else float('inf')
                if second > best * (1.0 + self.uncertainty_ratio):
                    break

        selected = min(predictions, key=lambda b: predictions[b].predicted_s)
        overhead = perf_counter() - start
        return selected, predictions, probes, overhead
