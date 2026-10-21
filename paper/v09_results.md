# AQX v0.9 Results and Verdict

## Executive result

AQX v0.9 was evaluated on 20 freshly measured benchmark cases spanning four quantitative-finance workload families and five problem sizes per family. Each backend was measured with one warmup execution followed by three steady-state timing repeats on the same host, using a fixed RNG seed for reproducibility.

The primary leave-one-size-rank-out experiment used three random seeds for the selector and produced 60 held-out predictions per method. The strongest lightweight selector (logistic regression on the numeric workload features) achieved 95.0% oracle-hit accuracy and 3.52% mean regret. A size-only ablation achieved 70.0% accuracy and 154.6% mean regret, showing that structural workload features materially improve backend selection.

For context, the global fixed-backend baseline selected from the training folds had 49.4% mean regret. The family-fixed baseline had 102.9% mean regret. Thus AQX's numeric selector reduced mean regret by approximately 92.9% relative to the global fixed baseline on this benchmark.

## Robustness and generalization

A deliberately difficult leave-one-family-out experiment removed an entire workload family from training. The numeric selector retained 80.0% accuracy with 12.48% mean regret and 71.06% p95 regret. This indicates that the model is not merely memorizing family labels and provides evidence that the feature representation captures some cross-family performance structure.

The confidence intervals remain material because the current benchmark contains only 20 cases. The 95% bootstrap interval for primary mean regret was 0.0% to 8.21%; for cross-family mean regret it was 5.98% to 19.60%.

## Ablation

The regularization ablation was stable enough to support a clear qualitative conclusion. With C=0.1, accuracy was 85.0% and mean regret 8.44%. With C=1.0, accuracy was 95.0% and mean regret 3.52%. With C=10.0, accuracy was 90.0% and mean regret 4.73%. The selected C=1.0 model therefore outperformed both more strongly regularized and more weakly regularized variants in this benchmark.

## What the experiment does and does not establish

The results support the hypothesis that measurable workload characteristics can predict which of several Python execution backends will perform best for these quantitative kernels. The result is especially strong relative to trivial fixed-backend baselines and to a size-only predictor.

The results do not yet establish general superiority over production-grade compiler/autotuning systems, nor do they establish performance portability across CPUs, Python distributions, operating systems, or hardware generations. The benchmark is still modest in size, the current backends are limited to Python/NumPy/Numba variants, and the measurements were collected on one host.

## Research novelty position

Machine-learning-based compiler autotuning is already a mature research area. Prior work has studied optimization selection, phase ordering, low-fidelity measurements, and learning-guided autotuning. AQX should therefore not be presented as a generic compiler autotuner. The defensible positioning is narrower: adaptive selection among Python execution backends for quantitative-finance workloads, with regret-aware evaluation and explicit cross-family generalization tests. This distinction is essential for a publication claim.

## Publication verdict

**Status: promising but not yet submission-ready.**

The v0.9 data is strong enough to justify continuing the project and drafting a paper. The central result is now credible enough to motivate a research contribution rather than a classroom optimization exercise. However, a reviewer would still demand a larger benchmark, multiple hardware environments, stronger comparisons to existing automatic optimization systems, and more exhaustive statistical validation.

The next research milestone should be v1.0: a 100+ case benchmark corpus, at least two machines, repeated-seed evaluation, confidence intervals reported at case level, and comparisons against stronger automatic baselines. The main success criterion should be sustained regret reduction versus the best fixed-backend baseline without excessive selection overhead.
