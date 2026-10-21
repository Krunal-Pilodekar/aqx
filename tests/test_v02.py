import numpy as np

from aqx.backends.multi import BACKENDS, get_backend, prepare_inputs


def test_backend_numerical_agreement():
    for workload, size in [("monte_carlo", 128), ("black_scholes", 128), ("rolling_signal", 256), ("portfolio_risk", 256)]:
        args, kwargs = prepare_inputs(workload, size)
        vals = [float(get_backend(workload, b)(*args, **kwargs)) for b in BACKENDS]
        assert np.allclose(vals, vals[0], rtol=1e-8, atol=1e-8), (workload, vals)


def test_all_backends_callable():
    for workload in ("monte_carlo", "black_scholes", "rolling_signal", "portfolio_risk"):
        args, kwargs = prepare_inputs(workload, 64)
        for backend in BACKENDS:
            value = get_backend(workload, backend)(*args, **kwargs)
            assert np.isfinite(float(value))
