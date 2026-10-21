# AQX v0.6 — Learning-to-Select Results

## Research question

Can a supervised model predict the fastest execution backend for quantitative-finance workloads at an unseen larger input size, while using an uncertainty threshold to abstain on weak predictions?

## Dataset and split

The experiment reuses the measured steady-state cases collected in AQX v0.5. No new timings are synthesized. There are four workload families and four measured sizes per family. For each family, the two smallest sizes are used for training, the third size is used only for validation, and the largest size is held out for final testing.

## Model

AQX v0.6 uses a Random Forest classifier over static workload features. The classifier estimates the probability that each backend is the fastest. Prediction margin and entropy form uncertainty signals. The abstention thresholds are tuned on the validation set and frozen before final testing. Abstention falls back to a fixed `numba` backend.

## Metrics

The principal deployment metric is regret: selected backend time divided by oracle time minus one. We also report oracle-hit accuracy, raw model accuracy before abstention, abstention rate, and improvement over the fixed fallback.

## Interpretation

A positive result requires AQX to reduce held-out regret relative to the fallback without relying on test-set threshold tuning. A negative result is still informative because it identifies whether static workload features are sufficient for generalization at larger problem sizes.

## Reproducibility note

The timing dataset is inherited from v0.5 and therefore reflects the same hardware/software environment and steady-state measurement procedure. This v0.6 experiment is a model-selection experiment on a fixed measured dataset; it does not rerun the benchmark kernels.
