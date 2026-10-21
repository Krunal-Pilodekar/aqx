from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; P=ROOT/'plots'; P.mkdir(exist_ok=True)
rows=json.loads((R/'v04_adaptive_selection.json').read_text())
workloads=sorted({r['workload'] for r in rows})
# Regret comparison
x=np.arange(len(rows)); width=0.38
fig=plt.figure(figsize=(10,5)); ax=fig.add_axes([0.08,0.16,0.88,0.76])
ax.bar(x-width/2,[100*r['static_regret'] for r in rows],width,label='Static')
ax.bar(x+width/2,[100*r['adaptive_regret'] for r in rows],width,label='Multi-fidelity')
ax.set_ylabel('Regret (%)'); ax.set_xlabel('Held-out case'); ax.set_title('AQX v0.4: Static vs multi-fidelity regret'); ax.set_xticks(x, [f"{r['workload']}\n{r['size']}" for r in rows],rotation=45,ha='right'); ax.legend(); ax.grid(axis='y',alpha=0.2)
fig.savefig(P/'v04_static_vs_adaptive_regret.png',dpi=160); plt.close(fig)
# Probe/overhead tradeoff
fig=plt.figure(figsize=(7,5)); ax=fig.add_axes([0.12,0.14,0.82,0.78])
ax.scatter([r['measurement_budget_cases'] for r in rows],[100*r['adaptive_regret'] for r in rows])
for r in rows: ax.annotate(r['workload'],(r['measurement_budget_cases'],100*r['adaptive_regret']),fontsize=8,xytext=(4,4),textcoords='offset points')
ax.set_xlabel('Number of pilot measurements'); ax.set_ylabel('Regret (%)'); ax.set_title('Measurement budget vs regret'); ax.grid(alpha=0.2)
fig.savefig(P/'v04_budget_vs_regret.png',dpi=160); plt.close(fig)
# per-workload summary
out={}
for w in workloads:
    rs=[r for r in rows if r['workload']==w]
    out[w]={
      'n':len(rs),
      'static_accuracy':float(np.mean([r['static_correct'] for r in rs])),
      'adaptive_accuracy':float(np.mean([r['adaptive_correct'] for r in rs])),
      'static_mean_regret':float(np.mean([r['static_regret'] for r in rs])),
      'adaptive_mean_regret':float(np.mean([r['adaptive_regret'] for r in rs])),
      'mean_calibration_s':float(np.mean([r['calibration_s'] for r in rs]))}
(R/'v04_per_workload.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
