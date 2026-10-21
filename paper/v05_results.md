# AQX v0.5 Preliminary Results

## Method

v0.5 introduces an uncertainty-aware selector. Instead of trusting a low-fidelity extrapolation, AQX measures only the top-ranked candidate backends, repeats each steady-state probe, computes robust uncertainty statistics, and applies multiplicative corrections to the static cost model. The selector can abstain and retain the static decision when evidence is insufficient.

One-time warm-up/JIT time is recorded separately from steady-state timing. This prevents compilation latency from being conflated with the throughput of the compiled backend.

## Research hypothesis

The testable hypothesis is that uncertainty-aware calibration should reduce harmful backend switches while using less measurement effort than exhaustive benchmarking.

## Result interpretation

The included experiment is deliberately treated as an evaluation, not a success claim. The final paper should report whether v0.5 improves mean/median regret and selection accuracy over the static selector, and how often it abstains. A positive result is only meaningful if calibration overhead and repeated-measurement cost are also reported.

## Limitations

The current benchmark suite is synthetic and executed on one machine. The selector is still a prototype and the cost model is trained on small numbers of cases. Publication-grade conclusions require larger benchmark matrices, more independent repetitions, multiple machines, held-out workload families, and statistical significance analysis.
