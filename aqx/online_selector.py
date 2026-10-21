from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calibration import OnlineCalibrator, CalibrationPrediction
from .cost_model import AdaptiveCostModel
from .backends.multi import BACKENDS
from .workload import WorkloadSpec


@dataclass
class OnlineSelection:
    selected: str
    static_choice: str
    predictions: dict[str, float]
    calibrated_predictions: dict[str, float]
    calibration_s: float
    candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "static_choice": self.static_choice,
            "predictions": self.predictions,
            "calibrated_predictions": self.calibrated_predictions,
            "calibration_s": self.calibration_s,
            "candidates": list(self.candidates),
        }


class AdaptiveOnlineSelector:
    """Hybrid selector: static model narrows candidates, online calibration resolves the choice."""

    def __init__(self, static_model: AdaptiveCostModel, top_k: int = 3, pilot_points: int = 2, repeats: int = 2):
        self.static_model = static_model
        self.top_k = max(1, min(int(top_k), len(BACKENDS)))
        self.calibrator = OnlineCalibrator(BACKENDS, pilot_points=pilot_points, repeats=repeats)

    def select(self, spec: WorkloadSpec, seed: int = 7) -> OnlineSelection:
        static_predictions = self.static_model.predict(spec)
        ranked = sorted(static_predictions, key=static_predictions.get)
        candidates = tuple(ranked[: self.top_k])
        static_choice = ranked[0]
        selected, predictions, _, overhead = self.calibrator.select(spec.name, spec.size, candidates, seed=seed)
        calibrated = {b: p.predicted_execute_s for b, p in predictions.items()}
        return OnlineSelection(selected, static_choice, static_predictions, calibrated, overhead, candidates)
