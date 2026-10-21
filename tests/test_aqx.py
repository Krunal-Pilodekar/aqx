import numpy as np
from aqx.backends.monte_carlo import generate_shocks, mc_python, mc_numpy, mc_numba, mc_numba_parallel
from aqx.features import extract_mc_features
from aqx.selector import RuleSelector, oracle

def test_backends_numerically_agree():
    z = generate_shocks(200, 20, 11)
    vals = [mc_python(z), mc_numpy(z), mc_numba(z), mc_numba_parallel(z)]
    assert max(vals) - min(vals) < 1e-8

def test_selector_returns_known_backend():
    z = generate_shocks(1000, 20)
    sel = RuleSelector().select(extract_mc_features(z))
    assert sel.backend in {'numpy','numba','numba_parallel'}

def test_oracle():
    assert oracle({'a': 3.0, 'b': 2.0, 'c': 4.0}) == 'b'
