# AQX Preliminary Results

## Status

This document records an exploratory result from the first AQX prototype. It is **not** a publication claim.

## Experiment

The experiment uses a deterministic Monte Carlo European call-option kernel. A matrix of standard-normal shocks is generated once and passed to every backend so that numerical outputs are directly comparable. Backends are evaluated with five repeated timing runs after an explicit Numba warm-up, and the median execution time is used for comparisons.

The held-out selector is a simple log-linear cost model trained only on a separate subset of workload sizes. It predicts execution time for each backend and selects the lowest predicted cost. The oracle uses the measured held-out timing and chooses the fastest backend after the fact.

## Observed held-out behavior

| Elements | Oracle | AQX selection | Regret |
|---:|---|---|---:|
| 100,000 | Numba parallel | NumPy | 24.2% |
| 1,000,000 | Numba parallel | Numba parallel | 0.0% |
| 10,000,000 | Numba parallel | Numba parallel | 0.0% |

Selection accuracy is 2/3 on these three held-out cases. The median regret is 0%, and the worst-case regret is 24.2%.

For the 10,000,000-element case, the observed median execution time is approximately 0.0379 s for NumPy, 0.0632 s for Numba, and 0.0369 s for Numba-parallel; the CPython reference is approximately 1.266 s. All four implementations agree within roughly 4.1e-14 absolute error on the tested case.

## Interpretation

The preliminary experiment does not yet establish general superiority. It instead exposes an important modeling problem: one-dimensional workload size is insufficient to reliably predict the best backend near crossover points. A better AQX selector should incorporate at least path count, step count, memory footprint, dtype, and an explicit model of fixed backend overheads.

The most promising next experiment is an **online cost-aware selector** that performs a small calibration probe on the actual input (or representative sample), estimates each backend's total cost, and includes compilation/setup overhead before selecting a backend.

## Threats to validity

- Only one benchmark family has been evaluated.
- The current workload feature space is intentionally small.
- The tests use a single machine and one Python/NumPy/Numba environment.
- The held-out test set contains only three configurations.
- Numba warm-up is outside the timed execution benchmark; cold-start costs require a separate experiment.
- Parallel performance is sensitive to thread count and system load.

## Next research stage

1. Add four or more additional finance kernels.
2. Expand train/test sizes and use grouped cross-validation by workload family.
3. Add richer static and runtime features.
4. Add cold-start versus steady-state experiments.
5. Compare rule-based, learned, online-probe, and oracle selectors.
6. Add JAX and C++ baselines only after the Python-native baselines are stable.
