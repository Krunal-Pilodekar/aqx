# Contributing to AQX

AQX is an experimental research project. Contributions that improve correctness, reproducibility, benchmarks, documentation, or backend support are welcome.

## Development

```bash
python -m venv .venv
# activate the environment for your platform
pip install -e .
pytest -q
```

## Benchmark contributions

For benchmark changes, document the workload definition, input size, machine information, software versions, and measurement protocol. Do not replace failed measurements with estimates.

## Pull requests

Please keep changes focused, include tests where practical, and explain any change that affects benchmark methodology or research claims.
