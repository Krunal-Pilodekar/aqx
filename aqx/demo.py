"""Small user-facing AQX demo.

The demo deliberately performs a fresh measurement instead of loading research
results. It is intended to let a new user understand the core idea in <1 minute.
"""
from __future__ import annotations

import argparse
import statistics
import time

from .backends.multi import BACKENDS, get_backend, prepare_inputs


def _time_backend(workload: str, backend: str, size: int, repeats: int = 3) -> tuple[float, float]:
    args, kwargs = prepare_inputs(workload, size)
    fn = get_backend(workload, backend)
    # Warm-up once so compiled backends are not compared against compilation time.
    fn(*args, **kwargs)
    samples = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = float(fn(*args, **kwargs))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny AQX backend-selection demo.")
    parser.add_argument("--workload", choices=("monte_carlo", "black_scholes", "rolling_signal", "portfolio_risk"), default="monte_carlo")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.size <= 0 or args.repeats <= 0:
        raise SystemExit("size and repeats must be positive")

    rows = []
    reference = None
    for backend in BACKENDS:
        t, value = _time_backend(args.workload, backend, args.size, args.repeats)
        if reference is None:
            reference = value
        rows.append((backend, t, value))

    fastest = min(rows, key=lambda r: r[1])
    print(f"AQX demo | workload={args.workload} | size={args.size}")
    print("-" * 58)
    print(f"{'backend':<18}{'median seconds':>18}{'speedup vs slowest':>20}")
    slowest = max(r[1] for r in rows)
    for backend, t, _ in sorted(rows, key=lambda r: r[1]):
        print(f"{backend:<18}{t:>18.6f}{slowest / t:>20.2f}x")
    print("-" * 58)
    print(f"Measured fastest backend: {fastest[0]}")
    print("Numerical result agreement is expected across backends.")


if __name__ == "__main__":
    main()
