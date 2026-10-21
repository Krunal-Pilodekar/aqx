"""First AQX prototype: tiny cost-aware backend interface.

This intentionally does not claim to be the final selector. It provides
an executable structure that later experiments can replace with learned
selection logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Any


@dataclass
class RunResult:
    backend: str
    compile_or_setup_s: float
    execute_s: float
    total_s: float
    value: Any


def timed_run(name: str, fn: Callable[[], Any]) -> RunResult:
    t0 = perf_counter()
    value = fn()
    total = perf_counter() - t0
    return RunResult(name, 0.0, total, total, value)


def choose_fastest(results: list[RunResult]) -> RunResult:
    if not results:
        raise ValueError("No backend results supplied")
    return min(results, key=lambda r: r.total_s)
