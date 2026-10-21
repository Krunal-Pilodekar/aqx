from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .workload import WorkloadSpec
from .backends.multi import BACKENDS


def _feature_vector(spec: WorkloadSpec) -> np.ndarray:
    return np.array([
        1.0,
        np.log1p(spec.elements),
        np.log1p(spec.bytes),
        np.log1p(spec.parallelism),
        np.log1p(spec.arithmetic_intensity),
        float(spec.dimensions),
        float(spec.dtype_size),
    ], dtype=float)


@dataclass
class BackendModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def predict_log(self, x: np.ndarray) -> float:
        z = (x[1:] - self.mean) / self.scale
        return float(self.coef[0] + np.dot(self.coef[1:], z))


class AdaptiveCostModel:
    """Simple, interpretable per-backend cost model with ridge regularization."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.models: dict[str, BackendModel] = {}

    def fit(self, rows: list[dict]) -> None:
        for backend in BACKENDS:
            usable = [r for r in rows if float(r.get(f"{backend}_total_s", r.get("medians", {}).get(backend, 0))) > 0]
            if len(usable) < 3:
                continue
            X = np.vstack([_feature_vector(WorkloadSpec(**r["spec"])) for r in usable])
            y = np.log(np.array([float(r.get(f"{backend}_total_s", r["medians"][backend])) for r in usable]))
            mean = X[:, 1:].mean(axis=0)
            scale = X[:, 1:].std(axis=0)
            scale[scale == 0] = 1.0
            Z = np.column_stack([np.ones(len(X)), (X[:, 1:] - mean) / scale])
            reg = np.eye(Z.shape[1]) * self.alpha
            reg[0, 0] = 0.0
            coef = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
            self.models[backend] = BackendModel(mean, scale, coef)

    def predict(self, spec: WorkloadSpec) -> dict[str, float]:
        x = _feature_vector(spec)
        return {b: float(np.exp(m.predict_log(x))) for b, m in self.models.items()}

    def select(self, spec: WorkloadSpec) -> str:
        predictions = self.predict(spec)
        if not predictions:
            raise RuntimeError("Cost model has not been fitted")
        return min(predictions, key=predictions.get)
