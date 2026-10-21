# AQX User Guide

## Install

After AQX is published to PyPI:

```bash
python -m pip install aqx
```

For a local checkout:

```bash
python -m pip install -e .
```

For research/development dependencies, run `python -m pip install -e ".[research,dev]"`.

## Run the quick demo

```bash
aqx-demo --workload monte_carlo --size 4096
```

Use another workload with `--workload black_scholes`, `rolling_signal`, or `portfolio_risk`.

## What the demo means

The demo is an empirical comparator: it warms each backend, measures several steady-state runs, and reports the fastest backend on your machine for that workload. It does not claim to train or deploy the research selector. That separation is intentional so new users can reproduce the basic performance question without relying on stored benchmark results.

## Research experiments

The `experiments/` directory contains versioned runners and analysis scripts. The `results/` directory contains measured or derived artifacts. Do not overwrite those files when adding new hardware results; create a new run identifier and record the environment.

## Reporting a new hardware result

Include CPU model, number of cores, OS, Python version, NumPy version, Numba version, AQX commit hash, workload, size, number of repeats, warm-up policy, and the full command.
