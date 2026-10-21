from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from aqx.backends.multi import BACKENDS
from aqx.cost_model import AdaptiveCostModel
from aqx.evaluator import evaluate_case
from aqx.adaptive_v04 import MultiFidelitySelector
from aqx.workload import WorkloadSpec

WORKLOADS = {
    'monte_carlo': [512, 2048, 8192, 16384],
    'black_scholes': [512, 2048, 8192, 16384],
    'rolling_signal': [1024, 4096, 16384, 32768],
    'portfolio_risk': [512, 2048, 8192, 16384],
}
RESULTS=ROOT/'results'; PLOTS=ROOT/'plots'; RESULTS.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)

def main():
    cases=[]; raw=[]
    for w,sizes in WORKLOADS.items():
        for s in sizes:
            meas,spec=evaluate_case(w,s,repeats=3)
            med={b:float(np.median([m.execute_s for m in meas if m.backend==b])) for b in BACKENDS}
            oracle=min(med,key=med.get)
            cases.append({'workload':w,'size':s,'spec':spec.to_dict(),'medians':med,'oracle':oracle})
            raw.extend(m.to_dict() for m in meas)
    train=[]; test=[]
    for w in WORKLOADS:
        fam=[c for c in cases if c['workload']==w]
        train.extend(fam[:2]); test.extend(fam[2:])
    model=AdaptiveCostModel(alpha=0.25); model.fit(train)
    selector=MultiFidelitySelector(model,top_k=3,max_probes=2,uncertainty_ratio=0.15,repeats=1,seed=7)
    rows=[]
    for c in test:
        spec=WorkloadSpec(**c['spec'])
        sel=selector.select(spec)
        oracle_t=c['medians'][c['oracle']]
        st=c['medians'][sel.static_selected]; at=c['medians'][sel.selected]
        rows.append({
            'workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'static_selected':sel.static_selected,
            'adaptive_selected':sel.selected,'static_correct':sel.static_selected==c['oracle'],'adaptive_correct':sel.selected==c['oracle'],
            'static_regret':st/oracle_t-1.0,'adaptive_regret':at/oracle_t-1.0,
            'oracle_time_s':oracle_t,'adaptive_selected_time_s':at,'calibration_s':sel.calibration_s,
            'measurement_budget_cases':sum(sel.probe_counts.values()),'probe_counts':sel.probe_counts,
            'candidates':list(sel.candidates),'decision_reason':sel.decision_reason,
            'static_predictions':sel.static_predictions,'calibrated_predictions':sel.calibrated_predictions,'probes':sel.probes
        })
    test_cases_by_key={(c['workload'], c['size']): c for c in test}
    exhaustive_cost=float(sum(sum(test_cases_by_key[(r['workload'], r['size'])]['medians'].values()) for r in rows))
    adaptive_cost=float(sum(r['calibration_s'] for r in rows))
    summary={
        'version':'0.4','n_total_cases':len(cases),'n_test':len(test),'top_k':selector.top_k,
        'max_probes':selector.calibrator.max_probes,'uncertainty_ratio':selector.calibrator.uncertainty_ratio,
        'adaptive_selection_accuracy':float(np.mean([r['adaptive_correct'] for r in rows])),
        'static_selection_accuracy':float(np.mean([r['static_correct'] for r in rows])),
        'adaptive_mean_regret':float(np.mean([r['adaptive_regret'] for r in rows])),
        'static_mean_regret':float(np.mean([r['static_regret'] for r in rows])),
        'adaptive_median_regret':float(np.median([r['adaptive_regret'] for r in rows])),
        'adaptive_max_regret':float(np.max([r['adaptive_regret'] for r in rows])),
        'mean_calibration_s':float(np.mean([r['calibration_s'] for r in rows])),
        'median_calibration_s':float(np.median([r['calibration_s'] for r in rows])),
        'mean_probe_count':float(np.mean([r['measurement_budget_cases'] for r in rows])),
        'estimated_exhaustive_benchmark_cost_s':exhaustive_cost,
        'adaptive_calibration_cost_s':adaptive_cost,
        'estimated_budget_fraction_vs_full_oracle':adaptive_cost/max(exhaustive_cost,1e-12),
        'fraction_regret_improved_vs_static':float(np.mean([r['adaptive_regret']<r['static_regret'] for r in rows]))
    }
    (RESULTS/'v04_cases.json').write_text(json.dumps(cases,indent=2))
    (RESULTS/'v04_adaptive_selection.json').write_text(json.dumps(rows,indent=2))
    (RESULTS/'v04_summary.json').write_text(json.dumps(summary,indent=2))
    with (RESULTS/'v04_raw.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=raw[0].keys()); wr.writeheader(); wr.writerows(raw)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
