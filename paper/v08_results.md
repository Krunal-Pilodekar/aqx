# AQX v0.8 — Risk-Aware Backend Selection

## Objective

AQX v0.8 tests whether a runtime selector can improve over probability-only backend classification by penalizing predicted runtime uncertainty. The selector fits one random-forest runtime model per backend and uses the empirical tree prediction dispersion as an uncertainty signal. Backend choice minimizes predicted runtime plus a tunable uncertainty penalty.

## Evaluation protocol

The timing dataset is the fixed AQX v0.5 measured steady-state dataset: 16 cases spanning Monte Carlo simulation, Black–Scholes pricing, rolling/statistical signals, and portfolio risk. No synthetic timings were introduced.

The primary evaluation uses an outer leave-one-size-rank-out split across workload families. The risk penalty was tuned only on inner validation folds. Five random seeds were used for the final outer evaluation. The final test set therefore remains unseen during hyperparameter selection.

## Results

| Metric | AQX v0.8 | AQX v0.7 | Always Numba |
|---|---:|---:|---:|
| Oracle-hit accuracy | 77.5% | 77.5% | 75.0%* |
| Mean regret | **18.69%** | 35.95% | 39.32% |
| Median regret | 0.0% | 0.0% | 0.0% |
| P95 regret | **67.22%** | 452.03% | not primary |
| Maximum regret | 309.76% | 452.03% | not primary |
| Win rate vs Numba | 20.0% | 27.5% | — |

\* Always-Numba oracle-hit rate is included only as a descriptive comparison over the same case set; the primary baseline is its regret.

The v0.8 risk-aware objective substantially reduces mean regret relative to v0.7 while preserving the same mean oracle-hit accuracy. The largest observed regret also decreases, although a 309.76% outlier remains.

The 95% bootstrap confidence interval for v0.8 mean regret is 7.03%–33.00%; for accuracy it is 67.5%–86.25%.

## Interpretation

The result supports the hypothesis that predicted runtime uncertainty contains useful information beyond the most likely backend label. A risk-aware decision rule can avoid some high-cost mistakes without reducing top-1 backend selection accuracy.

However, the result does **not** establish universal superiority. AQX still loses to the fixed Numba baseline on many individual cases, and the remaining maximum regret shows that the model can still make severe errors. The benchmark is also deliberately small because the underlying measured dataset contains only 16 cases.

## Research implication

The next experiment should enlarge the timing corpus substantially and test whether the reduction in regret survives across machines, repeated measurements, additional workloads, and unseen workload families. A useful ablation should compare: classifier-only selection, mean-runtime regression, risk-aware selection, and a conservative abstention policy.
