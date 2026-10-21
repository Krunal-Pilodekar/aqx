from __future__ import annotations
import json, sys, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aqx.workload import WorkloadSpec
from aqx.ml_selector import BackendClassifier

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
BACKENDS=('python','numpy','numba','numba_parallel')
FALLBACK='numba'

# Reuse v0.5's measured steady-state case matrix exactly as a fixed benchmark dataset.
cases=json.loads((RESULTS/'source_v05_cases.json').read_text())

def split_cases(cases):
    train=[]; val=[]; test=[]
    for w in sorted({c['workload'] for c in cases}):
        fam=sorted([c for c in cases if c['workload']==w], key=lambda x:x['size'])
        train += fam[:2]; val += fam[2:3]; test += fam[3:4]
    return train,val,test

def eval_rows(model, records, threshold, entropy_threshold):
    out=[]
    for c in records:
        pred=model.predict(WorkloadSpec(**c['spec']), threshold=threshold, entropy_threshold=entropy_threshold)
        fallback=c['medians'][FALLBACK]
        selected=FALLBACK if pred.abstained else pred.selected
        selected_t=c['medians'][selected]; oracle_t=c['medians'][c['oracle']]
        out.append({'workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':pred.selected,
                    'selected':selected,'raw_model_correct':pred.selected==c['oracle'],'correct':selected==c['oracle'],
                    'regret':selected_t/oracle_t-1.0,'fallback_regret':fallback/oracle_t-1.0,
                    'margin':pred.margin,'entropy':pred.entropy,'abstained':pred.abstained,'reason':pred.reason,
                    'probabilities':pred.probabilities,'oracle_time_s':oracle_t,'selected_time_s':selected_t})
    return out

def mean(xs): return float(np.mean(xs)) if xs else float('nan')

train,val,test=split_cases(cases)
model=BackendClassifier(n_estimators=500,random_state=17,min_leaf=1); model.fit(train)
# Validation-only threshold selection. Objective favors low regret, then fewer abstentions.
# Cache validation probabilities once, then tune thresholds without repeated model inference.
val_pred=[]
for c in val:
    pred=model.predict(WorkloadSpec(**c["spec"]), threshold=0.0, entropy_threshold=999.0)
    val_pred.append((c,pred))
opts=[]
for t in np.linspace(0,0.6,13):
    for e in np.linspace(0.2,1.4,13):
        rr=[]
        for c,pred in val_pred:
            abst=pred.margin < float(t) or pred.entropy > float(e)
            sel=FALLBACK if abst else pred.selected
            regret=c['medians'][sel]/c['medians'][c['oracle']]-1.0
            rr.append((regret,abst))
        score=mean([x[0] for x in rr]) + 0.005*mean([x[1] for x in rr])
        opts.append((score,float(t),float(e)))
opts.sort(key=lambda x:x[0])
_,threshold,entropy_threshold=opts[0]
val_rows=[]
for c,pred in val_pred:
    abst=pred.margin < threshold or pred.entropy > entropy_threshold
    sel=FALLBACK if abst else pred.selected
    val_rows.append({"workload":c["workload"],"size":c["size"],"oracle":c["oracle"],"predicted":pred.selected,"selected":sel,
        "correct":sel==c["oracle"],"raw_model_correct":pred.selected==c["oracle"],"regret":c["medians"][sel]/c["medians"][c["oracle"]]-1.0,
        "abstained":abst,"margin":pred.margin,"entropy":pred.entropy,"probabilities":pred.probabilities,"reason":pred.reason})
test_rows=eval_rows(model,test,threshold,entropy_threshold)
# Always-numba baseline on test.
fallback_regrets=[x['fallback_regret'] for x in test_rows]
summary={
 'version':'0.6','source_dataset':'AQX v0.5 measured steady-state cases','n_total_cases':len(cases),
 'n_train':len(train),'n_validation':len(val),'n_test':len(test),'repeats_source':3,
 'split':'2 train sizes / 1 validation size / 1 held-out test size per workload',
 'model':'RandomForestClassifier','n_estimators':500,'min_samples_leaf':1,'random_state':17,
 'tuned_margin_threshold':threshold,'tuned_entropy_threshold':entropy_threshold,
 'validation_accuracy':mean([x['correct'] for x in val_rows]),
 'validation_mean_regret':mean([x['regret'] for x in val_rows]),
 'test_accuracy':mean([x['correct'] for x in test_rows]),
 'test_raw_model_accuracy':mean([x['raw_model_correct'] for x in test_rows]),
 'test_mean_regret':mean([x['regret'] for x in test_rows]),
 'test_median_regret':float(np.median([x['regret'] for x in test_rows])),
 'test_max_regret':float(np.max([x['regret'] for x in test_rows])),
 'test_abstention_fraction':mean([x['abstained'] for x in test_rows]),
 'fallback_mean_regret':mean(fallback_regrets),
 'regret_improvement_vs_fallback':mean(fallback_regrets)-mean([x['regret'] for x in test_rows]),
 'model_train_log_loss':model.train_log_loss,
}
for name,obj in [('v06_train.json',train),('v06_validation.json',val),('v06_test.json',test_rows),('v06_summary.json',summary)]:
    (RESULTS/name).write_text(json.dumps(obj,indent=2))
(RESULTS/'v06_model_config.json').write_text(json.dumps({'threshold':threshold,'entropy_threshold':entropy_threshold,
    'fallback':FALLBACK,'source_dataset':'AQX_v0.5/results/v05_cases.json'},indent=2))
print(json.dumps(summary,indent=2))
