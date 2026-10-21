# AQX

**Adaptive Quantitative Python Execution**

AQX is a research prototype for **risk-aware adaptive backend selection** in quantitative Python workloads. It studies whether a system can choose among existing execution strategies—rather than asking developers to manually benchmark every implementation—while controlling the risk of a bad choice.

> **Research status:** experimental / pre-publication. Results in this repository are evidence from a small benchmark corpus, not production performance guarantees.

## Why AQX?

Quantitative Python code can run through multiple execution paths: ordinary Python, NumPy vectorization, Numba JIT, and Numba parallel execution. The fastest option depends on workload structure, input size, compilation effects, and machine characteristics.

AQX treats backend selection as a **prediction + risk-control problem**.

```text
Python workload
      |
      v
workload features
      |
      v
backend-selection model
      |
      +--> uncertainty / regret risk
      |
      v
select backend or use safe fallback
```

## Current research result

The v0.9 measured corpus contains **20 cases** across four quantitative-finance workload families and five sizes, with measurements for four backends: pure Python, NumPy, Numba, and Numba-parallel.

Under the repository's primary held-out protocol, the best lightweight selector reached:

- **95.0% oracle-hit accuracy**
- **3.52% mean regret**
- **0% median regret**

A deliberately harder leave-one-workload-family-out experiment reached **80.0% accuracy** and **12.48% mean regret**.

These are promising results, but the benchmark currently uses one machine and a small corpus. The project is therefore not yet presented as a production optimizer or as definitive evidence of general performance.

## Installation

Python 3.10+ is supported.

After the first PyPI release, install the published distribution with:

```bash
python -m pip install aqx-adaptive-execution
```

To install the current source checkout instead:

```bash
python -m pip install -e .
```

For local research and development dependencies:

```bash
python -m pip install -e ".[research,dev]"
```

The distribution name (`aqx`) must be available on PyPI at publication time. If PyPI rejects it as unavailable or reserved, choose a distinct name such as `aqx-quant` in `pyproject.toml`; the Python import name can remain `aqx`.

## Tests

```bash
pytest -q
```

## Reproduce the v0.9 analysis

```bash
python experiments/run_v09_clean.py
```

The repository keeps measured timing data under `results/` and the research interpretation under `paper/`.

## Project structure

```text
aqx/                 # selector and execution components
benchmarks/          # benchmark definitions
experiments/         # experiment runners and analyses
results/             # measured / derived experiment artifacts
plots/               # research figures
paper/               # research notes and results
tests/               # correctness and regression tests
```

## Research positioning

AQX does **not** claim to invent JIT compilation, vectorization, or automatic parallelization. Those areas have extensive prior art. The research question is narrower:

> Can a risk-aware selector choose among existing Python execution backends for quantitative workloads while approaching an oracle policy with substantially less manual backend selection?

## Roadmap

- v0.9 — robustness, ablation, cross-family generalization
- v1.0 — larger benchmark corpus, additional hardware, stronger external baselines, decision-overhead accounting
- future — optional C++ / GPU backends if the selection problem remains scientifically compelling

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Benchmark improvements are especially valuable when they include reproducible measurement protocols and machine metadata.

## License

MIT. See [`LICENSE`](LICENSE).

## 60-second demo

Install the repository in editable mode, then run:

```bash
pip install -e .
aqx-demo --workload monte_carlo --size 4096
```

The demo warms each backend before measuring it, reports median steady-state time, and shows the empirically fastest backend for that machine and workload. It is deliberately separate from the learned research selector so users can reproduce the basic performance question without downloading trained models.

## v1.0 research-release scope

v1.0 freezes the current research baseline and makes the repository easier for outside users to reproduce and challenge. The strongest current result comes from a 20-case corpus and a leave-one-size-rank-out protocol. It should be treated as a research finding, not as a production guarantee.

The next publication-grade experiment should expand the corpus, repeat measurements across hardware, compare stronger autotuning baselines, and report end-to-end selector overhead.

## Contributing benchmark evidence

If you benchmark AQX on another machine, please include Python, NumPy, Numba, CPU model, OS, workload, input size, and the exact command used. This lets the project distinguish algorithmic effects from hardware-specific behavior.
