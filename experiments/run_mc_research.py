from __future__ import annotations

import json
from pathlib import Path
import csv
import numpy as np
from numba import set_num_threads

from aqx.backends.monte_carlo import generate_shocks, mc_python, mc_numpy, mc_numba, mc_numba_parallel
from aqx.features import extract_mc_features
from aqx.runner import benchmark
from aqx.selector import oracle
from aqx.learned_selector import LearnedSelector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
BACKENDS = {'python': mc_python, 'numpy': mc_numpy, 'numba': mc_numba, 'numba_parallel': mc_numba_parallel}

# Train/test are deliberately separated. The selector never sees test timings during fitting.
CONFIGS = [
    (500, 10), (500, 50), (1_000, 25), (2_000, 50), (5_000, 100),
    (10_000, 25), (10_000, 100), (20_000, 50), (50_000, 25),
    (50_000, 50), (100_000, 50), (100_000, 100),
]
TRAIN = CONFIGS[::2] + CONFIGS[1::4]
TEST = [c for c in CONFIGS if c not in TRAIN]


def collect(configs):
    rows = []
    for paths, steps in configs:
        z = generate_shocks(paths, steps, seed=7)
        f = extract_mc_features(z)
        timings = {}
        values = {}
        for name, fn in BACKENDS.items():
            rs = benchmark(lambda fn=fn: fn(z), repeats=5, warmup=0,
                           workload='monte_carlo', backend=name, paths=paths, steps=steps)
            timings[name] = float(np.median([r.execute_s for r in rs]))
            values[name] = rs[0].value
        ref = values['python']
        row = {'paths': paths, 'steps': steps, 'elements': int(z.size), **{f'{b}_median_s': timings[b] for b in BACKENDS}}
        row['oracle'] = oracle(timings)
        row['max_abs_error'] = max(abs(v-ref) for v in values.values())
        row['features'] = f.to_dict()
        rows.append(row)
    return rows


def main():
    set_num_threads(max(1, min(8, __import__('os').cpu_count() or 1)))
    warm = generate_shocks(100, 10); mc_numba(warm); mc_numba_parallel(warm)
    train = collect(TRAIN)
    model = LearnedSelector(); model.fit(train)
    test = collect(TEST)
    eval_rows = []
    for row in test:
        f = extract_mc_features(generate_shocks(row['paths'], row['steps'], seed=7))
        pred = model.predict(f)
        selected = model.select(f)
        best_t = row[f"{row['oracle']}_median_s"]
        selected_t = row[f"{selected}_median_s"]
        row = dict(row)
        row.update({'selected': selected, 'selection_correct': selected == row['oracle'],
                    'regret': (selected_t-best_t)/best_t, 'predicted_times_s': pred})
        eval_rows.append(row)
    (OUT/'mc_research.json').write_text(json.dumps({'train':train,'test':eval_rows}, indent=2))
    print(json.dumps(eval_rows, indent=2))

if __name__ == '__main__': main()
