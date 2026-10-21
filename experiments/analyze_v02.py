from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
P = ROOT / "plots"

with open(R / "v02_test_selection.json") as f:
    rows = json.load(f)
with open(R / "v02_summary.json") as f:
    summary = json.load(f)

# Per-workload table.
workloads = sorted({r["workload"] for r in rows})
per = []
for w in workloads:
    x = [r for r in rows if r["workload"] == w]
    per.append({
        "workload": w,
        "n": len(x),
        "accuracy": float(np.mean([r["selection_correct"] for r in x])),
        "mean_regret": float(np.mean([r["regret"] for r in x])),
        "median_regret": float(np.median([r["regret"] for r in x])),
        "max_regret": float(np.max([r["regret"] for r in x])),
    })
with open(R / "v02_per_workload.json", "w") as f:
    json.dump(per, f, indent=2)

# Plot 1: oracle backend frequencies.
freq = {b: sum(r["oracle"] == b for r in rows) for b in ["python", "numpy", "numba", "numba_parallel"]}
plt.figure(figsize=(8, 4.5))
plt.bar(freq.keys(), freq.values())
plt.ylabel("Held-out cases")
plt.title("Oracle backend frequency")
plt.tight_layout()
plt.savefig(P / "v02_oracle_frequency.png", dpi=160)
plt.close()

# Plot 2: AQX regret by workload.
plt.figure(figsize=(8, 4.5))
plt.bar([x["workload"] for x in per], [100*x["mean_regret"] for x in per])
plt.ylabel("Mean regret (%)")
plt.title("AQX mean regret by workload")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(P / "v02_regret_by_workload.png", dpi=160)
plt.close()

# Plot 3: predicted vs actual log runtime for all selected/test rows and backends.
points = []
for r in rows:
    pred = r.get("predictions", {})
    for b, p in pred.items():
        actual = r["medians"][b]
        points.append((p, actual))
px = np.array([x[0] for x in points]); py = np.array([x[1] for x in points])
plt.figure(figsize=(5.5, 5.5))
plt.loglog(px, py, "o", alpha=0.75)
lo = min(px.min(), py.min()); hi = max(px.max(), py.max())
plt.loglog([lo, hi], [lo, hi], "--")
plt.xlabel("Predicted runtime (s)")
plt.ylabel("Measured median runtime (s)")
plt.title("AQX cost model: prediction vs measurement")
plt.tight_layout()
plt.savefig(P / "v02_prediction_vs_actual.png", dpi=160)
plt.close()

print(json.dumps({"summary": summary, "per_workload": per}, indent=2))
