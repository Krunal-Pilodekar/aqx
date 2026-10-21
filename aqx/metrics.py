from __future__ import annotations

import numpy as np


def summarize_selector(rows: list[dict]) -> dict:
    regrets = np.array([float(r['regret']) for r in rows], dtype=float)
    correct = np.array([bool(r['selection_correct']) for r in rows], dtype=bool)
    return {
        'n': int(len(rows)),
        'selection_accuracy': float(correct.mean()) if len(rows) else None,
        'median_regret': float(np.median(regrets)) if len(rows) else None,
        'mean_regret': float(regrets.mean()) if len(rows) else None,
        'max_regret': float(regrets.max()) if len(rows) else None,
        'oracle_hits': int(correct.sum()),
    }
