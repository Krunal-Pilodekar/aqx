from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class WorkloadFeatures:
    paths: int
    steps: int
    elements: int
    bytes: int
    dimensions: int
    dtype_size: int
    arithmetic_intensity_proxy: float
    parallelism_proxy: float

    def to_dict(self):
        return asdict(self)


def extract_mc_features(z: np.ndarray) -> WorkloadFeatures:
    paths, steps = z.shape
    elements = int(z.size)
    bytes_ = int(z.nbytes)
    # Coarse proxies intentionally simple and interpretable in v1.
    arithmetic_proxy = 8.0  # exp/multiply/add/payoff operations per element, approximate.
    parallelism_proxy = float(paths)
    return WorkloadFeatures(
        paths=int(paths), steps=int(steps), elements=elements,
        bytes=bytes_, dimensions=int(z.ndim), dtype_size=int(z.dtype.itemsize),
        arithmetic_intensity_proxy=arithmetic_proxy,
        parallelism_proxy=parallelism_proxy,
    )
