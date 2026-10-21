from __future__ import annotations

import json
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
PLOT = ROOT / 'plots'; PLOT.mkdir(exist_ok=True)


def main():
    data = json.loads((RES/'mc_research.json').read_text())
    rows = data['test']
    metrics = {
        'selector_accuracy': sum(r['selection_correct'] for r in rows)/len(rows),
        'median_regret': sorted(r['regret'] for r in rows)[len(rows)//2],
        'max_regret': max(r['regret'] for r in rows),
        'oracle_hits': sum(r['selection_correct'] for r in rows),
    }
    (RES/'mc_metrics.json').write_text(json.dumps(metrics, indent=2))

    # Speedup chart vs CPython for held-out tests.
    xs = [r['elements'] for r in rows]
    for backend in ['numpy','numba','numba_parallel']:
        ys = [r['python_median_s']/r[f'{backend}_median_s'] for r in rows]
        plt.figure(figsize=(7,4.5))
        plt.plot(xs, ys, marker='o', label=backend)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Input elements (log scale)')
        plt.ylabel('Speedup vs CPython (log scale)')
        plt.title(f'Monte Carlo speedup: {backend} vs CPython')
        plt.grid(True, alpha=.25); plt.legend(); plt.tight_layout()
        plt.savefig(PLOT/f'speedup_{backend}.png', dpi=180); plt.close()

    plt.figure(figsize=(7,4.5))
    plt.plot(xs, [100*r['regret'] for r in rows], marker='o')
    plt.xscale('log'); plt.xlabel('Input elements (log scale)')
    plt.ylabel('AQX regret vs oracle (%)')
    plt.title('Held-out selector regret')
    plt.grid(True, alpha=.25); plt.tight_layout()
    plt.savefig(PLOT/'selector_regret.png', dpi=180); plt.close()

    print(json.dumps(metrics, indent=2))

if __name__ == '__main__': main()
