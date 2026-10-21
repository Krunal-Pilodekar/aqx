from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
res=json.loads((ROOT/'results/v06_summary.json').read_text()); rows=json.loads((ROOT/'results/v06_test.json').read_text())
plt.figure(figsize=(7,4.5))
plt.bar(['Fallback regret','AQX regret'], [100*res['fallback_mean_regret'],100*res['test_mean_regret']])
plt.ylabel('Mean regret (%)'); plt.title('AQX v0.6 held-out regret vs fallback'); plt.tight_layout(); plt.savefig(ROOT/'plots/v06_regret_vs_fallback.png',dpi=160); plt.close()
plt.figure(figsize=(7,4.5))
x=[r['margin'] for r in rows]; y=[100*r['regret'] for r in rows]
plt.scatter(x,y)
for r in rows: plt.annotate(f"{r['workload']}:{r['size']}",(r['margin'],100*r['regret']),fontsize=7)
plt.xlabel('Prediction probability margin'); plt.ylabel('Regret (%)'); plt.title('Model confidence vs held-out regret'); plt.tight_layout(); plt.savefig(ROOT/'plots/v06_confidence_regret.png',dpi=160); plt.close()
print(json.dumps(res,indent=2))
