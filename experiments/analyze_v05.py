from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
PLOTS = ROOT / 'plots'

def main():
    rows = json.loads((RESULTS/'v05_selection.json').read_text())
    summary = json.loads((RESULTS/'v05_summary.json').read_text())
    labels = [f"{r['workload']}\n{r['size']}" for r in rows]
    x = np.arange(len(rows))
    static = np.array([r['static_regret']*100 for r in rows])
    adaptive = np.array([r['adaptive_regret']*100 for r in rows])
    plt.figure(figsize=(10,5))
    plt.plot(x, static, marker='o', label='Static')
    plt.plot(x, adaptive, marker='o', label='AQX v0.5')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Regret vs. oracle (%)')
    plt.title('AQX v0.5: static vs uncertainty-aware regret')
    plt.legend()
    plt.tight_layout(); plt.savefig(PLOTS/'v05_static_vs_uncertainty_regret.png', dpi=180); plt.close()

    reasons = {}
    for r in rows: reasons[r['reason']] = reasons.get(r['reason'],0)+1
    plt.figure(figsize=(7,4))
    plt.bar(list(reasons), list(reasons.values()))
    plt.ylabel('Held-out cases')
    plt.title('AQX v0.5 decision outcomes')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout(); plt.savefig(PLOTS/'v05_decision_outcomes.png', dpi=180); plt.close()

    md = [
        '# AQX v0.5 Experiment Report', '',
        '## Outcome', '',
        f"Static selection accuracy: **{summary['static_selection_accuracy']:.1%}**.",
        f"AQX v0.5 selection accuracy: **{summary['adaptive_selection_accuracy']:.1%}**.",
        f"Static mean regret: **{summary['static_mean_regret']:.2%}**.",
        f"AQX v0.5 mean regret: **{summary['adaptive_mean_regret']:.2%}**.",
        f"Calibration used on **{summary['calibration_fraction']:.1%}** of held-out cases; abstention rate was **{summary['abstention_fraction']:.1%}**.", '',
        '## Interpretation', '',
        'The current policy is conservative and avoided harmful backend switches, but it did not improve the measured baseline. This means the confidence rule is too strict or the static model features are too weak to support useful calibration decisions at the present benchmark sizes.', '',
        '## Required next experiment', '',
        'Increase the training matrix, add repeated-run noise estimates to the training data, learn a probability of backend superiority, and optimize the abstention threshold on validation data only. The final test set must remain untouched until the policy is frozen.',
    ]
    (RESULTS/'v05_experiment_report.md').write_text('\n'.join(md))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
