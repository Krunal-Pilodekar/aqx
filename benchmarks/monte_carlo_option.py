from __future__ import annotations

import math
import time
import numpy as np


def black_scholes_mc_python(paths: int, steps: int, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.20, 1.0
    dt = t / steps
    total = 0.0
    for _ in range(paths):
        s = s0
        for _ in range(steps):
            z = rng.standard_normal()
            s *= math.exp((r - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
        total += max(s - k, 0.0)
    return math.exp(-r * t) * total / paths


def black_scholes_mc_numpy(paths: int, steps: int, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.20, 1.0
    dt = t / steps
    z = rng.standard_normal((paths, steps))
    increments = (r - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    terminal = s0 * np.exp(increments.sum(axis=1))
    payoff = np.maximum(terminal - k, 0.0)
    return math.exp(-r * t) * float(payoff.mean())


if __name__ == "__main__":
    for paths, steps in [(2_000, 50), (10_000, 50)]:
        for name, fn in [
            ("python", black_scholes_mc_python),
            ("numpy", black_scholes_mc_numpy),
        ]:
            t0 = time.perf_counter()
            value = fn(paths, steps)
            print(f"{paths=} {steps=} {name:>6} {time.perf_counter()-t0:.6f}s value={value:.6f}")
