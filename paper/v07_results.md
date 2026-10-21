# AQX v0.7 — Robustness and Paper-Readiness Assessment

## Executive verdict

AQX is **research-worthy but not yet publication-ready as a performance paper**. The current evidence supports the narrower claim that a workload-feature-based classifier can generalize backend choices across unseen size ranks on this small quantitative-finance benchmark, but it does **not** yet support a strong claim of practical performance superiority over simple baselines.

## Experimental protocol

The v0.7 analysis uses the fixed, measured AQX v0.5 steady-state dataset: 16 cases covering four workload families and four input-size ranks per family. No timings were synthesized. We use leave-one-size-rank-out validation: one size rank from every workload family is held out, the remaining three ranks are used for training, and the procedure is repeated across five Random Forest seeds (7, 17, 29, 41, 73). This produces 80 model predictions over 16 held-out cases.

The primary model is a Random Forest classifier using workload identity plus static workload features. The selector is compared with fixed-backend baselines, especially always-Numba and workload-family fixed choices. Regret is defined as selected runtime / oracle runtime - 1.

## Results

| Metric | AQX RF | Always Numba | Family-fixed baseline |
|---|---:|---:|---:|
| Mean regret | 35.95% | 39.32% | 63.41% |
| Median regret | 0.00% | 0.00% | — |
| 95th percentile regret | 452.03% | — | — |
| Maximum regret | 452.03% | — | — |
| Oracle-hit accuracy | 77.50% | 75.00% | — |
| 95% bootstrap CI, mean regret | [14.0%, 62.3%] | [22.2%, 59.4%] | [11.0%, 133.2%] |

AQX improves average regret over always-Numba by about 3.36 percentage points, but it wins against Numba on only 27.5% of individual model predictions. The gain comes from recognizing workloads such as the rolling-signal family where NumPy becomes clearly superior at larger sizes; it is partially offset by occasional large errors, especially when the model selects NumPy for the smallest rolling-signal case or selects parallel Numba where serial Numba is optimal.

The worst observed AQX error is the 1,024-element rolling-signal case, where the model selects NumPy while serial Numba is the measured oracle; regret is about 452%. This is a critical limitation because average regret alone would obscure the deployment risk of a selector that can make severe mistakes.

## What the results establish

1. The backend-selection problem is nontrivial: the measured oracle changes by workload and size.
2. A supervised model can learn meaningful structure from workload features; 77.5% oracle-hit accuracy across repeated leave-one-size-rank-out predictions is encouraging.
3. Simple adaptive selection is not yet robust enough for production claims because tail regret remains large.
4. On this dataset, a fixed Numba policy is a surprisingly strong baseline. Any publication claim must therefore compare against it directly and explain when AQX adds value.

## What the results do not establish

The dataset is too small to support broad claims about Python generally, quantitative finance generally, or different hardware/software environments. It also does not demonstrate that AQX reduces real end-to-end application time once profiling, model inference, compilation, and data movement costs are included. The current benchmark contains four workload families and only four measured size ranks per family.

## Publication verdict

**Current status: promising student/research-project prototype; not yet ready for submission as a systems-performance paper.**

A stronger paper would need at least:

- a substantially larger and more diverse benchmark matrix;
- independent machines or hardware profiles;
- repeated fresh-process and warm-state measurements;
- stronger baselines, including modern compiler/JIT alternatives where applicable;
- an explicit cost model for decision overhead;
- confidence-aware fallback that controls tail regret;
- ablation studies showing which features and selector components matter;
- statistical tests and confidence intervals at the case level;
- a held-out workload-family experiment, not just held-out sizes;
- an open, reproducible benchmark harness and complete environment specification.

## Recommended v0.8 research direction

The most promising change is **risk-aware backend selection**. Instead of minimizing expected runtime alone, AQX should optimize a utility such as:

`expected_runtime + lambda * tail_risk + decision_overhead`

or use a constrained objective such as:

`minimize expected regret subject to P(regret > threshold) <= alpha`.

This would turn the observed failure mode into a direct research contribution: a backend selector designed specifically to avoid catastrophic choices near performance crossovers.
