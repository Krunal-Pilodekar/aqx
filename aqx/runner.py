from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
import gc
import os
from typing import Callable

import numpy as np

@dataclass
class BenchmarkResult:
    workload: str
    backend: str
    paths: int
    steps: int
    repeat: int
    compile_or_setup_s: float
    execute_s: float
    total_s: float
    value: float
    peak_rss_mb: float | None

    def to_dict(self):
        return asdict(self)


def _rss_mb() -> float | None:
    try:
        import resource
        # Linux reports KiB; macOS bytes.
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / 1024.0 if os.uname().sysname == 'Linux' else raw / (1024.0 * 1024.0)
    except Exception:
        return None


def benchmark(fn: Callable[[], float], *, setup: Callable[[], object] | None = None, repeats=5, warmup=1,
              workload='unknown', backend='unknown', paths=0, steps=0) -> list[BenchmarkResult]:
    data = []
    if setup is not None:
        setup0 = perf_counter()
        setup()
        setup_s = perf_counter() - setup0
    else:
        setup_s = 0.0

    for _ in range(warmup):
        gc.collect()
        fn()

    for repeat in range(repeats):
        gc.collect()
        t0 = perf_counter()
        value = float(fn())
        execute_s = perf_counter() - t0
        data.append(BenchmarkResult(workload, backend, paths, steps, repeat,
                                    setup_s if repeat == 0 else 0.0,
                                    execute_s, setup_s + execute_s, value, _rss_mb()))
    return data
