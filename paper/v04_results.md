# AQX v0.4 Preliminary Results

## Scope

v0.4 tests a multi-fidelity selector that begins with a cheap probe for each shortlisted backend and spends a second probe only when the leading predictions are close or the fitted scaling curve is uncertain.

The experiment uses four quantitative-finance workload families with two training sizes and two held-out sizes per family. The current candidate set is pure Python, NumPy, Numba, and Numba-parallel.

## Results

The current run produced 50.0% static selection accuracy and 37.5% multi-fidelity selection accuracy on eight held-out cases. Mean regret increased from 16.25% for the static selector to 74.00% for the current multi-fidelity selector; median regret was 30.35% and maximum regret was 307.04%.

The result is **not treated as a success**. The experiment demonstrates that the present probe policy can make backend selection worse, likely because low-fidelity measurements do not preserve the target-size ranking reliably enough for these implementations. In particular, JIT and parallel backends can have crossover behavior that a simple log-log scaling fit cannot capture.

Mean decision/calibration time was approximately 0.56 seconds per held-out case in this environment. The next iteration therefore needs a better measurement policy and a more explicit treatment of compilation and warm-up effects.

## Research implication

The v0.4 contribution should not be stated as “multi-fidelity probing improves selection.” The defensible claim at this stage is narrower: adaptive probing is a promising direction, but the first policy is insufficient and reveals a concrete methodological problem—**low-fidelity execution rank is not necessarily a reliable proxy for target-size execution rank**.

## Next methodological changes

1. Separate compilation/setup effects from steady-state execution explicitly.
2. Use repeated probes and confidence intervals rather than a single median pilot.
3. Use rank-preserving validation across multiple probe sizes before trusting extrapolation.
4. Add an abstention/fallback rule: when uncertainty remains high, keep the static selection rather than forcing a calibrated choice.
5. Benchmark on larger problem sizes so backend crossover behavior is measurable above timer noise.

These changes are required before drawing publication-level conclusions.
