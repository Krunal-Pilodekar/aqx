from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class RobustStats:
    median_s: float
    mad_s: float
    ci_low_s: float
    ci_high_s: float
    cv: float


def bootstrap_median_ci(values: list[float], seed: int = 7, n_boot: int = 2000, alpha: float = 0.05) -> RobustStats:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or np.any(~np.isfinite(x)):
        raise ValueError("values must be finite and non-empty")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    mean_abs = float(np.mean(np.abs(x)))
    cv = float(np.std(x, ddof=1) / max(np.mean(x), 1e-12)) if x.size > 1 else 0.0
    if x.size == 1:
        lo = hi = med
    else:
        rng = np.random.default_rng(seed)
        sample_idx = rng.integers(0, x.size, size=(n_boot, x.size))
        boot = np.median(x[sample_idx], axis=1)
        lo = float(np.quantile(boot, alpha / 2))
        hi = float(np.quantile(boot, 1 - alpha / 2))
    return RobustStats(med, mad, lo, hi, cv)


def interval_overlap_ratio(a: RobustStats, b: RobustStats) -> float:
    lo = max(a.ci_low_s, b.ci_low_s)
    hi = min(a.ci_high_s, b.ci_high_s)
    overlap = max(0.0, hi - lo)
    span = max(a.ci_high_s, b.ci_high_s) - min(a.ci_low_s, b.ci_low_s)
    return float(overlap / max(span, 1e-12))


def normalized_width(s: RobustStats) -> float:
    return float((s.ci_high_s - s.ci_low_s) / max(s.median_s, 1e-12))
