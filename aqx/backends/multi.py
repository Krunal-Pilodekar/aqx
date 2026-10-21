from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numba import njit, prange


def _mc_python(shocks: np.ndarray) -> float:
    paths, steps = shocks.shape
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.20, 1.0
    dt = t / steps
    total = 0.0
    for i in range(paths):
        s = s0
        for j in range(steps):
            z = shocks[i, j]
            s *= math.exp((r - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
        total += max(s - k, 0.0)
    return math.exp(-r * t) * total / paths


def _mc_numpy(shocks: np.ndarray) -> float:
    steps = shocks.shape[1]
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.20, 1.0
    dt = t / steps
    terminal = s0 * np.exp(((r - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * shocks).sum(axis=1))
    return float(math.exp(-r * t) * np.maximum(terminal - k, 0.0).mean())


@njit(cache=True)
def _mc_numba(shocks: np.ndarray) -> float:
    paths, steps = shocks.shape
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.20, 1.0
    dt = t / steps
    total = 0.0
    sqrt_dt = math.sqrt(dt)
    for i in range(paths):
        s = s0
        for j in range(steps):
            s *= math.exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * shocks[i, j])
        total += max(s - k, 0.0)
    return math.exp(-r * t) * total / paths


@njit(cache=True, parallel=True)
def _mc_numba_parallel(shocks: np.ndarray) -> float:
    paths, steps = shocks.shape
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.20, 1.0
    dt = t / steps
    payoffs = np.empty(paths, dtype=np.float64)
    sqrt_dt = math.sqrt(dt)
    for i in prange(paths):
        s = s0
        for j in range(steps):
            s *= math.exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * shocks[i, j])
        payoffs[i] = max(s - k, 0.0)
    return math.exp(-r * t) * payoffs.mean()


def _bs_price_python(x: np.ndarray) -> float:
    total = 0.0
    for s in x:
        d1 = (math.log(s / 100.0) + (0.05 + 0.5 * 0.20**2)) / 0.20
        d2 = d1 - 0.20
        nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        total += s * nd1 - 100.0 * math.exp(-0.05) * nd2
    return total / len(x)


def _bs_price_numpy(x: np.ndarray) -> float:
    d1 = (np.log(x / 100.0) + (0.05 + 0.5 * 0.20**2)) / 0.20
    d2 = d1 - 0.20
    erf = np.vectorize(math.erf)
    nd1 = 0.5 * (1.0 + erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + erf(d2 / math.sqrt(2.0)))
    return float(np.mean(x * nd1 - 100.0 * math.exp(-0.05) * nd2))


@njit(cache=True)
def _bs_price_numba(x: np.ndarray) -> float:
    total = 0.0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i in range(x.size):
        d1 = (math.log(x[i] / 100.0) + (0.05 + 0.5 * 0.20**2)) / 0.20
        d2 = d1 - 0.20
        nd1 = 0.5 * (1.0 + math.erf(d1 * inv_sqrt2))
        nd2 = 0.5 * (1.0 + math.erf(d2 * inv_sqrt2))
        total += x[i] * nd1 - 100.0 * math.exp(-0.05) * nd2
    return total / x.size


@njit(cache=True, parallel=True)
def _bs_price_numba_parallel(x: np.ndarray) -> float:
    out = np.empty(x.size, dtype=np.float64)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i in prange(x.size):
        d1 = (math.log(x[i] / 100.0) + (0.05 + 0.5 * 0.20**2)) / 0.20
        d2 = d1 - 0.20
        nd1 = 0.5 * (1.0 + math.erf(d1 * inv_sqrt2))
        nd2 = 0.5 * (1.0 + math.erf(d2 * inv_sqrt2))
        out[i] = x[i] * nd1 - 100.0 * math.exp(-0.05) * nd2
    return out.mean()


def _rolling_python(x: np.ndarray, window: int) -> float:
    acc = 0.0
    for i in range(window, x.size):
        s = 0.0
        for j in range(i - window, i):
            s += x[j]
        acc += x[i] - s / window
    return acc


def _rolling_numpy(x: np.ndarray, window: int) -> float:
    c = np.cumsum(np.insert(x, 0, 0.0))
    means = (c[window: x.size] - c[: x.size - window]) / window
    return float(np.sum(x[window:] - means))


@njit(cache=True)
def _rolling_numba(x: np.ndarray, window: int) -> float:
    acc = 0.0
    for i in range(window, x.size):
        s = 0.0
        for j in range(i - window, i):
            s += x[j]
        acc += x[i] - s / window
    return acc


@njit(cache=True, parallel=True)
def _rolling_numba_parallel(x: np.ndarray, window: int) -> float:
    out = np.empty(x.size - window, dtype=np.float64)
    for i in prange(window, x.size):
        s = 0.0
        for j in range(i - window, i):
            s += x[j]
        out[i - window] = x[i] - s / window
    return out.sum()


def _portfolio_python(returns: np.ndarray, cov: np.ndarray) -> float:
    assets = cov.shape[0]
    weights = [1.0 / assets] * assets
    var = 0.0
    for i in range(assets):
        for j in range(assets):
            var += weights[i] * cov[i, j] * weights[j]
    total = 0.0
    for row in returns:
        total += sum(row[i] * weights[i] for i in range(assets)) ** 2
    return var + total / len(returns)


def _portfolio_numpy(returns: np.ndarray, cov: np.ndarray) -> float:
    assets = cov.shape[0]
    w = np.full(assets, 1.0 / assets)
    pnl = returns @ w
    return float(w @ cov @ w + np.mean(pnl * pnl))


@njit(cache=True)
def _portfolio_numba(returns: np.ndarray, cov: np.ndarray) -> float:
    scenarios, assets = returns.shape
    weight = 1.0 / assets
    var = 0.0
    for i in range(assets):
        for j in range(assets):
            var += weight * cov[i, j] * weight
    total = 0.0
    for s in range(scenarios):
        pnl = 0.0
        for i in range(assets):
            pnl += returns[s, i] * weight
        total += pnl * pnl
    return var + total / scenarios


@njit(cache=True, parallel=True)
def _portfolio_numba_parallel(returns: np.ndarray, cov: np.ndarray) -> float:
    scenarios, assets = returns.shape
    weight = 1.0 / assets
    var = 0.0
    for i in range(assets):
        for j in range(assets):
            var += weight * cov[i, j] * weight
    total = 0.0
    for s in prange(scenarios):
        pnl = 0.0
        for i in range(assets):
            pnl += returns[s, i] * weight
        total += pnl * pnl
    return var + total / scenarios


BACKENDS = ("python", "numpy", "numba", "numba_parallel")


def prepare_inputs(workload: str, size: int, seed: int = 7) -> tuple[tuple, dict]:
    rng = np.random.default_rng(seed + size)
    if workload == "monte_carlo":
        return (rng.standard_normal((size, 8)),), {}
    if workload == "black_scholes":
        return (np.exp(rng.normal(math.log(100), 0.15, size)).astype(np.float64),), {}
    if workload == "rolling_signal":
        window = min(64, max(8, size // 64))
        return (rng.normal(size=size).astype(np.float64),), {"window": window}
    if workload == "portfolio_risk":
        assets = max(4, int(math.sqrt(size)))
        scenarios = max(16, size // assets)
        a = rng.normal(size=(scenarios, assets)).astype(np.float64)
        raw = rng.normal(size=(assets, assets)).astype(np.float64)
        cov = raw @ raw.T / assets
        return (a, cov), {}
    raise ValueError(workload)


def get_backend(workload: str, backend: str) -> Callable:
    table = {
        "monte_carlo": {"python": _mc_python, "numpy": _mc_numpy, "numba": _mc_numba, "numba_parallel": _mc_numba_parallel},
        "black_scholes": {"python": _bs_price_python, "numpy": _bs_price_numpy, "numba": _bs_price_numba, "numba_parallel": _bs_price_numba_parallel},
        "rolling_signal": {"python": _rolling_python, "numpy": _rolling_numpy, "numba": _rolling_numba, "numba_parallel": _rolling_numba_parallel},
        "portfolio_risk": {"python": _portfolio_python, "numpy": _portfolio_numpy, "numba": _portfolio_numba, "numba_parallel": _portfolio_numba_parallel},
    }
    return table[workload][backend]
