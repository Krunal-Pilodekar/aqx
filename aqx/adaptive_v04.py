from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

from .backends.multi import BACKENDS
from .cost_model import AdaptiveCostModel
from .multifidelity import MultiFidelityCalibrator
from .workload import WorkloadSpec

@dataclass
class V04Selection:
    selected: str
    static_selected: str
    candidates: tuple[str, ...]
    static_predictions: dict[str, float]
    calibrated_predictions: dict[str, float]
    calibration_s: float
    probe_counts: dict[str, int]
    probes: dict[str, list[dict[str, Any]]]
    decision_reason: str

    def to_dict(self):
        return asdict(self)

class MultiFidelitySelector:
    def __init__(self, model: AdaptiveCostModel, top_k: int = 3, max_probes: int = 2,
                 uncertainty_ratio: float = 0.15, repeats: int = 1, seed: int = 7):
        self.model = model
        self.top_k = max(1, min(int(top_k), len(BACKENDS)))
        self.calibrator = MultiFidelityCalibrator(
            BACKENDS, max_probes=max_probes, uncertainty_ratio=uncertainty_ratio,
            repeats=repeats, seed=seed
        )

    def select(self, spec: WorkloadSpec, seed: int = 7) -> V04Selection:
        static_predictions = self.model.predict(spec)
        ranked = tuple(sorted(static_predictions, key=static_predictions.get))
        candidates = ranked[: self.top_k]
        static_selected = candidates[0]
        selected, preds, probes, overhead = self.calibrator.select(spec.name, spec.size, candidates)
        calibrated = {b: p.predicted_s for b, p in preds.items()}
        counts = {b: len(probes[b]) for b in candidates}
        reason = "adaptive_probe" if any(v > 1 for v in counts.values()) else "single_probe_confident"
        return V04Selection(selected, static_selected, candidates, static_predictions, calibrated,
                            overhead, counts, {b: [p.to_dict() for p in probes[b]] for b in candidates}, reason)
