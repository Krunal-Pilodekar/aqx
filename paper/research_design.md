# AQX Research Design

## 1. Proposed contribution

AQX studies an adaptive execution layer for quantitative Python workloads. The system observes static and runtime workload features and selects among several established execution mechanisms rather than assuming one backend is always optimal.

The novelty target is therefore the **selection/adaptation problem in a finance-specific workload family**, not the invention of JIT compilation, vectorization, or parallel execution themselves.

## 2. Why this is not an easy novelty claim

Automatic Python parallelization already exists. Numba can automatically parallelize supported operations and fuse adjacent operations; it can also compile numerical Python to machine code. ALPyNA previously demonstrated automatic dependence analysis and GPU-kernel generation for Python loops. Parakeet explored JIT parallel acceleration across CPUs and GPUs. These establish that “automatically make Python run faster” by itself is not a new research contribution.

More generally, compiler autotuning literature has studied selecting optimization strategies and phases, including machine-learning-based approaches. Recent work continues to explore automated compiler tuning and learning-driven optimization selection.

AQX must therefore demonstrate a more specific gap: **whether an inexpensive, transparent runtime selector can learn or infer which established Python execution backend is best for heterogeneous quantitative-finance kernels, while accounting for compilation overhead and numerical fidelity.**

## 3. Research questions

### RQ1 — Performance
How much execution-time reduction can adaptive backend selection provide over a conventional Python baseline?

### RQ2 — Oracle gap
How close does AQX come to an oracle that is allowed to benchmark all candidate backends and choose the fastest one?

### RQ3 — Generalization
Can a selector trained/calibrated on some quantitative workloads generalize to unseen workloads and unseen input sizes?

### RQ4 — Overhead
Does the cost of profiling, feature extraction, prediction, and compilation offset the execution benefit on small workloads?

### RQ5 — Fidelity
Can AQX preserve numerical correctness within an explicitly defined tolerance across backends?

## 4. Hypotheses

H1: AQX significantly reduces wall-clock execution time versus the CPython reference for sufficiently large numerical workloads.

H2: AQX remains within a small percentage gap of the best manually selected backend across a diverse benchmark suite.

H3: Static/runtime workload features can predict backend choice better than a single fixed backend policy.

H4: A cost-aware policy that includes compilation overhead outperforms a policy optimized only for steady-state execution on mixed small/large workloads.

## 5. Candidate features

Static features:

- number of Python loops
- loop nesting depth
- branch density
- obvious array operations
- reduction patterns
- function size / AST node counts
- use of supported NumPy operations

Runtime features:

- input element count
- array dimensionality
- dtype
- memory footprint
- observed iteration count
- warm-up versus steady-state time
- measured CPU count

Derived features:

- estimated arithmetic intensity
- estimated parallelism
- bytes processed per element
- branch-to-operation ratio
- problem-size class

## 6. Selection approaches

Stage A: rule-based selector. Establishes a transparent baseline.

Stage B: supervised classifier/regressor. Predicts expected runtime or backend ranking from measured/static features.

Stage C: cost-aware selector. Predicts total time as:

`T_total = T_analysis + T_compile + T_execute`

The final selector should optimize total time for the user's actual invocation pattern rather than only steady-state throughput.

## 7. Baselines

At minimum:

1. Pure Python / CPython reference implementation.
2. Hand-vectorized NumPy implementation where applicable.
3. Numba `njit`.
4. Numba `parallel=True` where applicable.
5. Oracle selector: benchmark candidates and choose the fastest for that workload/input size.

Optional later baselines:

- C++ native implementation
- JAX/JIT
- CuPy / GPU path
- PyPy where compatible

## 8. Benchmark protocol

For every workload and backend:

1. Verify output equivalence.
2. Run warm-up separately from timed runs.
3. Record compilation time explicitly.
4. Run repeated measurements with the same process policy.
5. Record median and dispersion, not just the fastest run.
6. Repeat across several problem sizes.
7. Record hardware, OS, Python version and package versions.
8. Pin thread counts where appropriate.

Primary metrics:

- total wall-clock time
- steady-state execution time
- speedup over CPython
- speedup over best manual backend
- oracle gap
- selector accuracy / regret
- peak RSS
- compilation overhead
- numerical error

## 9. Statistical analysis

For repeated benchmark measurements, report medians and robust dispersion. Use paired comparisons because the same workload instance can be evaluated under each backend. Report confidence intervals where appropriate. Avoid cherry-picking favorable runs or problem sizes.

A useful selector metric is **regret**:

`regret = (T_AQX - T_oracle) / T_oracle`

Lower is better. Report both median regret and worst-case regret.

## 10. Most important experiment

The core experiment should compare:

`CPython vs best fixed backend vs AQX vs oracle`

across unseen workload/input-size combinations.

The strongest plausible outcome is not necessarily “AQX beats C++.” A credible claim would be that AQX approaches the performance of the best manually selected existing Python acceleration path without requiring the researcher to manually choose and tune the backend.

## 11. Threats to validity

- Benchmark results may be hardware-specific.
- Numba compilation and caching can change observed totals.
- Some finance workloads are memory-bound rather than compute-bound.
- Backend support differs across Python/NumPy/Numba versions.
- Synthetic workloads may not represent production research code.
- Overfitting the selector to the benchmark suite would invalidate generalization claims.

## 12. Publication standard

The project should not claim publication novelty until the final literature review identifies the nearest competing systems and the experiments show a measurable advantage or new insight over them.

The paper should make three contributions only if supported by results:

1. an explicit workload/backend selection method;
2. an open, reproducible finance benchmark suite;
3. an empirical characterization of when each execution strategy wins and how close adaptive selection comes to the oracle.

## 13. Initial source set

- Numba documentation, “Automatic parallelization with @jit.” https://numba.readthedocs.io/en/stable/user/parallel.html
- Jacob, Dejice & Singer, Jeremy. “ALPyNA: Acceleration of Loops in Python for Novel Architectures.” 2019. https://eprints.gla.ac.uk/188418/
- Rubinsteyn et al. “Parakeet: A Just-In-Time Parallel Accelerator for Python.” USENIX HotPar 2012. https://www.usenix.org/conference/hotpar12/workshop-program/presentation/rubinsteyn
- Ashouri et al. “A Survey on Compiler Autotuning using Machine Learning.” 2018. https://arxiv.org/abs/1801.04405
- Zhu, Hao & Chen. “Compiler Auto-tuning through Multiple Phase Learning.” 2023. https://arxiv.org/abs/2304.14908
- Pan et al. “Compiler-R1: Towards Agentic Compiler Auto-tuning with Reinforcement Learning.” 2025. https://arxiv.org/abs/2506.15701
- Duarte et al. “Benchmarking machine-learning software and hardware for quantitative economics.” Journal of Economic Dynamics and Control, 2020. https://doi.org/10.1016/j.jedc.2019.103796

## v0.2 Experimental Protocol

The v0.2 implementation evaluates four quantitative-finance workload families (Monte Carlo simulation, Black–Scholes pricing, rolling signal calculation, and portfolio risk). Each family is tested at four increasing input sizes with deterministic inputs generated from fixed seeds. Each backend receives a warmup and three timed repetitions; the median execution time is used for backend ranking. The oracle backend is the backend with the lowest measured median execution time for the case.

AQX is trained offline on the lower half of the sizes within each workload family and evaluated on the held-out larger sizes. The cost model is deliberately interpretable: it uses log-scaled size, memory footprint, parallelism, arithmetic-intensity proxy, dimensionality, and dtype size. The model predicts log runtime separately for each backend. Selection quality is reported using exact oracle-hit accuracy and normalized regret, where regret = selected runtime / oracle runtime − 1.

The protocol explicitly records failures and does not treat a positive result as required. This is important for publication credibility because backend crossover behavior, compilation effects, memory traffic, and hardware-specific behavior can all make a simple model unreliable.
