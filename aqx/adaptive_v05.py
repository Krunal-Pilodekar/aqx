from __future__ import annotations
from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any
import gc
import numpy as np

from .backends.multi import BACKENDS, get_backend, prepare_inputs
from .uncertainty import bootstrap_median_ci, interval_overlap_ratio, normalized_width, RobustStats
from .workload import WorkloadSpec, make_spec
from .cost_model import AdaptiveCostModel


def _pilot_sizes(target: int, fractions=(1/32, 1/16, 1/8)) -> list[int]:
    if target <= 64:
        return [target]
    vals = [max(64, int(round(target * f))) for f in fractions]
    vals = [min(target, v) for v in vals]
    return sorted(set(vals))


def _steady_measure(workload: str, backend: str, size: int, repeats: int, seed: int) -> tuple[RobustStats, float]:
    args, kwargs = prepare_inputs(workload, size, seed=seed)
    fn = get_backend(workload, backend)
    # Explicit warm-up isolates steady-state execution from one-time JIT compilation.
    t0 = perf_counter(); fn(*args, **kwargs); warmup_s = perf_counter() - t0
    vals: list[float] = []
    for _ in range(repeats):
        gc.collect()
        t1 = perf_counter(); fn(*args, **kwargs); vals.append(perf_counter() - t1)
    return bootstrap_median_ci(vals, seed=seed + size), float(warmup_s)

@dataclass
class CalibrationEvidence:
    backend: str
    size: int
    stats: dict[str, float]
    warmup_s: float
    ratio_to_static: float

    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass
class V05Selection:
    selected: str
    static_selected: str
    used_calibration: bool
    abstained: bool
    candidates: tuple[str, ...]
    corrected_predictions: dict[str, float]
    confidence: dict[str, float]
    evidence: list[dict[str, Any]]
    calibration_s: float
    decision_reason: str

    def to_dict(self): return asdict(self)

class UncertaintyAwareSelector:
    """Selectively calibrates top candidates and abstains when evidence is weak."""

    def __init__(self, model: AdaptiveCostModel, top_k: int = 2, repeats: int = 3,
                 margin_threshold: float = 0.18, max_cv: float = 0.10,
                 max_interval_width: float = 0.20, min_rank_stability: float = 0.67,
                 seed: int = 7):
        self.model = model
        self.top_k = max(1, min(int(top_k), len(BACKENDS)))
        self.repeats = int(repeats)
        self.margin_threshold = float(margin_threshold)
        self.max_cv = float(max_cv)
        self.max_interval_width = float(max_interval_width)
        self.min_rank_stability = float(min_rank_stability)
        self.seed = int(seed)

    def select(self, spec: WorkloadSpec) -> V05Selection:
        static = self.model.predict(spec)
        ranked = tuple(sorted(static, key=static.get))
        static_selected = ranked[0]
        candidates = ranked[:self.top_k]
        if len(candidates) == 1:
            return V05Selection(static_selected, static_selected, False, True, candidates, static, {}, [], 0.0, "single_candidate")
        static_margin = static[candidates[1]] / max(static[candidates[0]], 1e-12) - 1.0
        # If model strongly prefers a backend, avoid expensive calibration unless the model is historically uncertain.
        if static_margin > self.margin_threshold:
            return V05Selection(static_selected, static_selected, False, True, candidates, static, {}, [], 0.0, "abstain_static_margin")

        start = perf_counter()
        sizes = _pilot_sizes(spec.size)
        evidence: list[CalibrationEvidence] = []
        ratios: dict[str, list[float]] = {b: [] for b in candidates}
        conf: dict[str, list[float]] = {b: [] for b in candidates}
        rank_votes = {b: 0 for b in candidates}
        total_rank_checks = 0

        for size in sizes:
            measured: dict[str, RobustStats] = {}
            for b in candidates:
                stats, warmup_s = _steady_measure(spec.name, b, size, self.repeats, self.seed)
                baseline_scale = max(static[b], 1e-12)
                # Compare measured scaling to the static prediction at the same effective problem size.
                # Use the static model prediction ratio from target to pilot to form a dimensionless correction.
                pilot_spec = make_spec(spec.name, size)
                predicted_pilot = self.model.predict(pilot_spec).get(b, baseline_scale)
                ratio = stats.median_s / max(predicted_pilot, 1e-12)
                ratios[b].append(float(ratio))
                width = normalized_width(stats)
                conf[b].append(float(max(0.0, 1.0 - min(1.0, stats.cv / max(self.max_cv, 1e-12)))))
                evidence.append(CalibrationEvidence(b, size, {"median_s": stats.median_s, "mad_s": stats.mad_s,
                    "ci_low_s": stats.ci_low_s, "ci_high_s": stats.ci_high_s, "cv": stats.cv}, warmup_s, float(ratio)))
            # Measure empirical rank at this scale; it is evidence about whether backend ordering is stable.
            ordered = sorted(measured, key=lambda x: measured[x].median_s) if measured else []
            # Reconstruct from just-added evidence for current size.
            current = {b: next(e for e in reversed(evidence) if e.backend == b and e.size == size).stats["median_s"] for b in candidates}
            ordered = sorted(candidates, key=current.get)
            for rank, b in enumerate(ordered):
                if rank == 0:
                    rank_votes[b] += 1
            total_rank_checks += 1

        corrected = {}
        for b in candidates:
            correction = float(np.median(ratios[b]))
            correction = float(np.clip(correction, 0.25, 4.0))
            corrected[b] = static[b] * correction

        corrected_ranked = sorted(candidates, key=corrected.get)
        best = corrected_ranked[0]
        second = corrected_ranked[1]
        correction_gap = corrected[second] / max(corrected[best], 1e-12) - 1.0
        rank_stability = max(rank_votes.values()) / max(total_rank_checks, 1)
        evidence_conf = min(np.mean(conf[b]) for b in candidates)
        selected_confidently = (
            correction_gap >= self.margin_threshold / 2
            and rank_stability >= self.min_rank_stability
            and evidence_conf >= 0.75
        )
        if selected_confidently:
            selected = best; abstained = False; reason = "calibrated_confident"
        else:
            selected = static_selected; abstained = True; reason = "abstain_uncertain"
        overhead = perf_counter() - start
        return V05Selection(selected, static_selected, True, abstained, candidates, corrected,
                            {b: float(np.mean(conf[b])) for b in candidates},
                            [e.to_dict() for e in evidence], overhead, reason)
