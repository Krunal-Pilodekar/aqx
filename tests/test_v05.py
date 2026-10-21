import numpy as np
from aqx.uncertainty import bootstrap_median_ci, interval_overlap_ratio, normalized_width
from aqx.adaptive_v05 import _pilot_sizes

def test_bootstrap_stats_finite():
    s = bootstrap_median_ci([1.0, 1.1, 0.9, 1.0], n_boot=200)
    assert s.ci_low_s <= s.median_s <= s.ci_high_s
    assert s.cv >= 0

def test_pilot_sizes_are_bounded():
    xs = _pilot_sizes(10000)
    assert xs == sorted(set(xs))
    assert all(64 <= x <= 10000 for x in xs)

def test_interval_helpers():
    a = bootstrap_median_ci([1, 1, 1], n_boot=100)
    b = bootstrap_median_ci([2, 2, 2], n_boot=100)
    assert interval_overlap_ratio(a,b) == 0.0
    assert normalized_width(a) == 0.0
