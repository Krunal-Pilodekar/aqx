# AQX v0.3 — Online Calibration Results

## Contribution

AQX v0.3 adds an online calibration layer to the v0.2 adaptive selector. Instead of relying only on an offline workload-to-runtime model, the system executes each candidate backend on smaller pilot inputs, estimates a workload-specific scaling law, and predicts steady-state execution time at the target size.

## Experimental protocol

Four quantitative-finance workload families are evaluated: Monte Carlo simulation, Black–Scholes pricing, rolling signal calculation, and portfolio risk. Each family contains four sizes. The first two sizes are used for training the static cost model; the final two sizes are held out for selector evaluation. Online calibration uses two smaller pilot sizes for the top three candidates selected by the static model.

The oracle is defined post hoc as the backend with the lowest measured median execution time on the full-size target case. Regret is `(selected_time / oracle_time) - 1`. Calibration overhead is measured separately and is not hidden inside the execution-time comparison.

## Interpretation

The main question is whether online calibration improves selection quality relative to the v0.2-style static selector. A positive result requires lower regret and/or higher oracle-hit accuracy without an impractically large calibration cost.

The experiment should not be interpreted as proof of universal superiority. It is a controlled systems experiment on the implemented benchmark suite and hardware environment.

## Reproducibility

```bash
python -m pytest -q
python experiments/run_v03.py
python experiments/analyze_v03.py
```

Results are written to `results/v03_*` and plots to `plots/v03_*`.
