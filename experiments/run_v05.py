from __future__ import annotations
import json, csv, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from aqx.backends.multi import BACKENDS
from aqx.cost_model import AdaptiveCostModel
from aqx.evaluator import evaluate_case
from aqx.adaptive_v05 import UncertaintyAwareSelector
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
    selector=UncertaintyAwareSelector(model,top_k=2,repeats=3,margin_threshold=0.18,seed=7)
    rows=[]
    for c in test:
        spec=WorkloadSpec(**c['spec'])
        sel=selector.select(spec)
        oracle_t=c['medians'][c['oracle']]
        st=c['medians'][sel.static_selected]; at=c['medians'][sel.selected]
        rows.append({'workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'static_selected':sel.static_selected,
            'adaptive_selected':sel.selected,'static_correct':sel.static_selected==c['oracle'],'adaptive_correct':sel.selected==c['oracle'],
            'static_regret':st/oracle_t-1.0,'adaptive_regret':at/oracle_t-1.0,'oracle_time_s':oracle_t,
            'adaptive_selected_time_s':at,'calibration_s':sel.calibration_s,'used_calibration':sel.used_calibration,
            'abstained':sel.abstained,'reason':sel.decision_reason,'corrected_predictions':sel.corrected_predictions,
            'confidence':sel.confidence,'candidates':list(sel.candidates),'evidence':sel.evidence})
    summary={'version':'0.5','n_total_cases':len(cases),'n_test':len(test),
        'static_selection_accuracy':float(np.mean([r['static_correct'] for r in rows])),
        'adaptive_selection_accuracy':float(np.mean([r['adaptive_correct'] for r in rows])),
        'static_mean_regret':float(np.mean([r['static_regret'] for r in rows])),
        'adaptive_mean_regret':float(np.mean([r['adaptive_regret'] for r in rows])),
        'static_median_regret':float(np.median([r['static_regret'] for r in rows])),
        'adaptive_median_regret':float(np.median([r['adaptive_regret'] for r in rows])),
        'adaptive_max_regret':float(np.max([r['adaptive_regret'] for r in rows])),
        'mean_calibration_s':float(np.mean([r['calibration_s'] for r in rows])),
        'median_calibration_s':float(np.median([r['calibration_s'] for r in rows])),
        'calibration_fraction':float(np.mean([r['used_calibration'] for r in rows])),
        'abstention_fraction':float(np.mean([r['abstained'] for r in rows])),
        'regret_improved_fraction':float(np.mean([r['adaptive_regret'] < r['static_regret'] for r in rows])),
    }
    (RESULTS/'v05_cases.json').write_text(json.dumps(cases,indent=2))
    (RESULTS/'v05_selection.json').write_text(json.dumps(rows,indent=2))
    (RESULTS/'v05_summary.json').write_text(json.dumps(summary,indent=2))
    with (RESULTS/'v05_raw.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=raw[0].keys()); wr.writeheader(); wr.writerows(raw)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
