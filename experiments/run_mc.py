from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from numba import set_num_threads

from aqx.backends.monte_carlo import generate_shocks, mc_python, mc_numpy, mc_numba, mc_numba_parallel
from aqx.features import extract_mc_features
from aqx.runner import benchmark
from aqx.selector import RuleSelector, oracle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

BACKENDS = {
    'python': mc_python,
    'numpy': mc_numpy,
    'numba': mc_numba,
    'numba_parallel': mc_numba_parallel,
}


def main():
    set_num_threads(max(1, min(8, __import__('os').cpu_count() or 1)))
    configs = [(500, 25), (2_000, 50), (10_000, 50), (50_000, 50), (100_000, 100)]
    all_rows = []
    summary = []
    selector = RuleSelector()

    # Warm/compile native backends once outside the timed corpus.
    warm_z = generate_shocks(100, 10)
    mc_numba(warm_z); mc_numba_parallel(warm_z)

    for paths, steps in configs:
        z = generate_shocks(paths, steps, seed=7)
        f = extract_mc_features(z)
        per_backend = {}
        per_rows = []
        for name, fn in BACKENDS.items():
            rows = benchmark(lambda fn=fn: fn(z), repeats=5, warmup=0,
                             workload='monte_carlo', backend=name, paths=paths, steps=steps)
            per_rows.extend(rows)
            per_backend[name] = float(np.median([r.execute_s for r in rows]))
        selected = selector.select(f).backend
        orch = per_backend[selected]
        best = oracle(per_backend)
        best_t = per_backend[best]
        regret = (orch - best_t) / best_t
        value_ref = per_rows[0].value
        max_err = max(abs(r.value - value_ref) for r in per_rows)
        row = {
            'paths': paths, 'steps': steps, 'elements': int(z.size),
            'selected': selected, 'oracle': best, 'selector_regret': regret,
            'max_abs_error': max_err,
            **{f'{k}_median_s': v for k, v in per_backend.items()},
        }
        summary.append(row)
        all_rows.extend([r.to_dict() for r in per_rows])

    with (OUT / 'mc_raw.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=all_rows[0].keys())
        writer.writeheader(); writer.writerows(all_rows)
    with (OUT / 'mc_summary.json').open('w') as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
