from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None
    prange = range


def mc_python(z: np.ndarray, s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0) -> float:
    paths, steps = z.shape
    dt = t / steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)
    total = 0.0
    disc = math.exp(-r * t)
    for i in range(paths):
        s = s0
        for j in range(steps):
            s *= math.exp(drift + diffusion * float(z[i, j]))
        payoff = s - k
        if payoff > 0.0:
            total += payoff
    return disc * total / paths


def mc_numpy(z: np.ndarray, s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0) -> float:
    steps = z.shape[1]
    dt = t / steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)
    terminal = s0 * np.exp((drift + diffusion * z).sum(axis=1))
    return float(math.exp(-r * t) * np.maximum(terminal - k, 0.0).mean())


if njit is not None:
    @njit(cache=True, fastmath=True)
    def _mc_numba(z, s0, k, r, sigma, t):
        paths, steps = z.shape
        dt = t / steps
        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)
        total = 0.0
        for i in range(paths):
            s = s0
            for j in range(steps):
                s *= math.exp(drift + diffusion * z[i, j])
            payoff = s - k
            if payoff > 0.0:
                total += payoff
        return math.exp(-r * t) * total / paths

    @njit(parallel=True, cache=True, fastmath=True)
    def _mc_numba_parallel(z, s0, k, r, sigma, t):
        paths, steps = z.shape
        dt = t / steps
        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)
        total = 0.0
        for i in prange(paths):
            s = s0
            for j in range(steps):
                s *= math.exp(drift + diffusion * z[i, j])
            payoff = s - k
            if payoff > 0.0:
                total += payoff
        return math.exp(-r * t) * total / paths

    def mc_numba(z, s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0) -> float:
        return float(_mc_numba(z, s0, k, r, sigma, t))

    def mc_numba_parallel(z, s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0) -> float:
        return float(_mc_numba_parallel(z, s0, k, r, sigma, t))
else:  # pragma: no cover
    def mc_numba(z, *args, **kwargs):
        raise RuntimeError("Numba unavailable")

    def mc_numba_parallel(z, *args, **kwargs):
        raise RuntimeError("Numba unavailable")


def generate_shocks(paths: int, steps: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((paths, steps), dtype=np.float64)
