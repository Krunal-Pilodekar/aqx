# Publishing AQX

## Before publishing

1. Confirm that the distribution name `aqx` is available on PyPI. If it is not, change `project.name` in `pyproject.toml` to a distinct distribution name such as `aqx-quant`. The import package can remain `aqx`.
2. Replace the `<YOUR-USERNAME>` placeholder in `CITATION.cff` with the final GitHub account/organization name.
3. Update the repository URL and release version in project metadata when the repository URL is known.
4. Test from a clean virtual environment and run the test suite.

## Build and validate

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Publish to TestPyPI first

Create an account at https://test.pypi.org and create an API token. Then upload:

```bash
python -m twine upload --repository testpypi dist/*
```

Test installation in a clean virtual environment. Install `numpy` and `numba` from the main PyPI first, then install the AQX package from TestPyPI with `--no-deps` so dependency resolution cannot accidentally select packages from the test index:

```bash
python -m pip install numpy numba
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps aqx
```

Run `aqx-demo --workload monte_carlo --size 4096` and the installed package tests/examples.

## Publish to PyPI

Once TestPyPI checks pass, create an account at https://pypi.org, enable account security, and use either PyPI Trusted Publishing via GitHub Actions or a scoped PyPI API token with Twine. For an initial manual upload:

```bash
python -m twine upload dist/*
```

Never commit a PyPI API token to GitHub, a source file, or a shell-history-bearing script. If the package name changes, update all install instructions before rebuilding. A published release file cannot be replaced in-place; increment the version for a corrected release.
