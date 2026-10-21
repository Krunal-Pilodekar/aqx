import numpy as np
from aqx.multifidelity import _fit, _probe_sizes, Probe

def test_probe_sizes_increase_and_bound():
    xs=_probe_sizes(10000)
    assert xs==sorted(set(xs))
    assert all(1<=x<=10000 for x in xs)
    assert xs[-1] < 10000

def test_fit_scaling_reasonable():
    pts=[Probe('x',100,0.01,1),Probe('x',1000,0.1,1),Probe('x',10000,1.0,1)]
    p=_fit(pts,100000)
    assert 8.0 < p.predicted_s < 12.0
    assert 0.8 < p.slope < 1.2
