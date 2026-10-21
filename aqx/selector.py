from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from .features import WorkloadFeatures

@dataclass(frozen=True)
class Selection:
    backend: str
    reason: str

class RuleSelector:
    """Transparent v1 selector used as the non-learned baseline."""

    def select(self, f: WorkloadFeatures) -> Selection:
        # Small workloads: avoid JIT compilation and prefer NumPy if material.
        if f.elements < 50_000:
            return Selection("numpy", "small workload; avoid JIT compilation overhead")
        # Medium workloads: Numba JIT can beat pure Python without parallel overhead.
        if f.elements < 1_000_000:
            return Selection("numba", "medium numerical loop; JIT amortization expected")
        # Large workloads: parallel execution becomes attractive.
        return Selection("numba_parallel", "large independent path dimension; parallelism likely beneficial")


def oracle(results: dict[str, float]) -> str:
    return min(results, key=results.get)
