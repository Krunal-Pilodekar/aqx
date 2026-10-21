# AQX v0.2 Preliminary Results

The v0.2 benchmark contains 16 total cases across four quantitative-finance workload families, with eight held out for selector evaluation.

| Metric | Result |
|---|---:|
| Held-out cases | 8 |
| Oracle-hit accuracy | 75.0% |
| Median regret | 0.0% |
| Mean regret | 29.9% |
| Worst-case regret | 151.4% |
| Mean oracle speedup vs pure Python | 135.0× |

These numbers are preliminary and should not be presented as a final publication result. The most important observation is that the selector can often identify the fastest backend, but misses can be expensive. This establishes a concrete optimization target for v0.3: improve prediction near backend crossover regions and incorporate backend-specific compilation/dispatch costs.
