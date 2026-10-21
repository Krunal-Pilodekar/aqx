from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .features import WorkloadFeatures

BACKENDS = ("python", "numpy", "numba", "numba_parallel")

@dataclass
class LogLinearModel:
    coef: np.ndarray

    def predict(self, features: WorkloadFeatures) -> float:
        x = np.log10(max(features.elements, 1))
        # Intercept + log(size). Keep this deliberately simple and interpretable in v1.
        y = float(self.coef[0] + self.coef[1] * x)
        return 10.0 ** y


class LearnedSelector:
    """Offline-calibrated cost model; fitting is separate from selection."""

    def __init__(self):
        self.models: dict[str, LogLinearModel] = {}

    def fit(self, rows: list[dict]) -> None:
        by_backend: dict[str, list[tuple[float, float]]] = {b: [] for b in BACKENDS}
        for row in rows:
            n = float(row["elements"])
            for backend in BACKENDS:
                y = float(row[f"{backend}_median_s"])
                if n > 0 and y > 0:
                    by_backend[backend].append((np.log10(n), np.log10(y)))
        for backend, pts in by_backend.items():
            if len(pts) < 2:
                continue
            X = np.array([[1.0, x] for x, _ in pts], dtype=float)
            y = np.array([v for _, v in pts], dtype=float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.models[backend] = LogLinearModel(coef)

    def predict(self, features: WorkloadFeatures) -> dict[str, float]:
        return {b: m.predict(features) for b, m in self.models.items()}

    def select(self, features: WorkloadFeatures) -> str:
        pred = self.predict(features)
        if not pred:
            raise RuntimeError("Selector has not been fitted")
        return min(pred, key=pred.get)
