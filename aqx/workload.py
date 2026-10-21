from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    size: int
    elements: int
    bytes: int
    dimensions: int
    dtype_size: int
    arithmetic_intensity: float
    parallelism: float
    problem_family: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_spec(name: str, size: int) -> WorkloadSpec:
    name = name.lower()
    if size <= 0:
        raise ValueError("size must be positive")

    if name == "monte_carlo":
        paths = size
        steps = 8
        elements = paths * steps
        return WorkloadSpec(name, size, elements, elements * 8, 2, 8, 12.0, float(paths), "simulation",
                             {"paths": paths, "steps": steps})
    if name == "black_scholes":
        elements = size
        return WorkloadSpec(name, size, elements, elements * 8 * 3, 1, 8, 18.0, float(size), "pricing",
                             {})
    if name == "rolling_signal":
        window = min(64, max(8, size // 64))
        elements = size
        return WorkloadSpec(name, size, elements, elements * 8 * 2, 1, 8, 8.0, float(size), "signal",
                             {"window": window})
    if name == "portfolio_risk":
        assets = max(4, int(np.sqrt(size)))
        scenarios = max(16, size // assets)
        elements = assets * scenarios + assets * assets
        return WorkloadSpec(name, size, elements, elements * 8, 2, 8, 10.0, float(scenarios), "risk",
                             {"assets": assets, "scenarios": scenarios})
    raise ValueError(f"Unknown workload: {name}")
