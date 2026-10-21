from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
rows = json.loads((RESULTS / "v03_online_selection.json").read_text())
summary = json.loads((RESULTS / "v03_summary.json").read_text())

labels = [f"{r['workload']}\n{r['size']}" for r in rows]
x = np.arange(len(rows))
static_regret = [100 * r["static_regret"] for r in rows]
online_regret = [100 * r["online_regret"] for r in rows]

plt.figure(figsize=(10, 5))
width = 0.36
plt.bar(x - width / 2, static_regret, width, label="Static model regret")
plt.bar(x + width / 2, online_regret, width, label="Online calibrated regret")
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Regret vs oracle (%)")
plt.title("AQX v0.3: Static vs Online-Calibrated Selector")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "v03_static_vs_online_regret.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(summary["mean_calibration_s"], summary["online_mean_regret"])
plt.xlabel("Mean calibration overhead (s)")
plt.ylabel("Online mean regret (%)")
plt.title("Calibration Cost / Selection Quality")
plt.tight_layout()
plt.savefig(PLOTS / "v03_calibration_tradeoff.png", dpi=160)
plt.close()

workloads = sorted(set(r["workload"] for r in rows))
static_by = [100 * np.mean([r["static_regret"] for r in rows if r["workload"] == w]) for w in workloads]
online_by = [100 * np.mean([r["online_regret"] for r in rows if r["workload"] == w]) for w in workloads]
plt.figure(figsize=(9, 5))
plt.bar(np.arange(len(workloads)) - width/2, static_by, width, label="Static")
plt.bar(np.arange(len(workloads)) + width/2, online_by, width, label="Online")
plt.xticks(np.arange(len(workloads)), workloads, rotation=20)
plt.ylabel("Mean regret (%)")
plt.title("Mean Regret by Workload")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "v03_regret_by_workload.png", dpi=160)
plt.close()

print(json.dumps({"summary": summary, "rows": rows}, indent=2))
