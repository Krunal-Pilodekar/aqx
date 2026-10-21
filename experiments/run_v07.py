from __future__ import annotations
import json, sys, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aqx.ml_selector import feature_row_from_record

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
cases=json.loads((RESULTS/'source_v05_cases.json').read_text())
BACKENDS=('python','numpy','numba','numba_parallel')

def prep(rows):
    X=pd.DataFrame([feature_row_from_record(r) for r in rows])
    y=np.array([r['oracle'] for r in rows])
    return X,y

def fit_model(rows, seed):
    X,y=prep(rows)
    num=[c for c in X.columns if c!='workload']; cat=['workload']
    pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num','passthrough',num)])
    clf=RandomForestClassifier(n_estimators=400,random_state=seed,min_samples_leaf=1,class_weight='balanced_subsample',n_jobs=-1)
    model=Pipeline([('pre',pre),('clf',clf)])
    model.fit(X,y)
    return model

def predict(model, row):
    X=pd.DataFrame([feature_row_from_record(row)])
    p=model.predict_proba(X)[0]
    classes=list(model.named_steps['clf'].classes_)
    probs={c:float(v) for c,v in zip(classes,p)}
    best=max(probs,key=probs.get)
    return best,probs

def regret(c,b):
    return c['medians'][b]/c['medians'][c['oracle']]-1.0

# Primary: leave-one-size-rank-out. Each fold holds out one measured size rank across ALL workload families.
by_w={w:sorted([c for c in cases if c['workload']==w], key=lambda z:z['size']) for w in sorted({c['workload'] for c in cases})}
fold_rows=[]
for rank in range(4):
    train=[]; test=[]
    for w,fam in by_w.items():
        test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
    for seed in [7,17,29,41,73]:
        model=fit_model(train,seed)
        for c in test:
            pred,probs=predict(model,c)
            fold_rows.append({
                'protocol':'leave_one_size_rank_out','heldout_rank':rank,'seed':seed,
                'workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':pred,
                'correct':pred==c['oracle'],'regret':regret(c,pred),'numba_regret':regret(c,'numba'),
                'numpy_regret':regret(c,'numpy'),'oracle_time_s':c['medians'][c['oracle']],
                'selected_time_s':c['medians'][pred],'probabilities':probs
            })

# Baselines and robust aggregation.
df=pd.DataFrame(fold_rows)
rf_metrics={
    'accuracy_mean':float(df.correct.mean()),
    'accuracy_std':float(df.groupby('seed').correct.mean().std(ddof=1)),
    'mean_regret':float(df.regret.mean()),
    'median_regret':float(df.regret.median()),
    'p95_regret':float(df.regret.quantile(0.95)),
    'max_regret':float(df.regret.max()),
    'mean_numba_regret':float(df.numba_regret.mean()),
    'mean_numpy_regret':float(df.numpy_regret.mean()),
    'win_rate_vs_numba':float((df.regret < df.numba_regret).mean()),
    'win_rate_vs_numpy':float((df.regret < df.numpy_regret).mean()),
}
# Workload-family best-fixed baseline chosen from training portion of each fold.
family_fixed=[]; global_fixed=[]
for rank in range(4):
    train=[]; test=[]
    for w,fam in by_w.items():
        test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
    global_counts=Counter(x['oracle'] for x in train); global_backend=global_counts.most_common(1)[0][0]
    for c in test:
        fam_train=[x for x in train if x['workload']==c['workload']]
        fam_backend=Counter(x['oracle'] for x in fam_train).most_common(1)[0][0]
        family_fixed.append({'heldout_rank':rank,'workload':c['workload'],'size':c['size'],'backend':fam_backend,'regret':regret(c,fam_backend)})
        global_fixed.append({'heldout_rank':rank,'workload':c['workload'],'size':c['size'],'backend':global_backend,'regret':regret(c,global_backend)})
ff=pd.DataFrame(family_fixed); gf=pd.DataFrame(global_fixed)

# Bootstrap CIs over the 80 model predictions (5 seeds x 16 cases); also case-level bootstraps for seed-averaged predictions.
rng=np.random.default_rng(123)
def bootstrap_mean(vals, n=10000):
    a=np.asarray(vals,float); means=[]
    for _ in range(n): means.append(float(rng.choice(a,size=len(a),replace=True).mean()))
    return float(np.quantile(means,0.025)),float(np.quantile(means,0.975))

summary={
 'version':'0.7',
 'dataset':'AQX v0.5 measured steady-state cases (fixed; no synthetic timings)',
 'n_cases':len(cases),'n_model_predictions':len(df),'seeds':[7,17,29,41,73],
 'protocol':'leave-one-size-rank-out across all workload families; one held-out rank per fold',
 'rf':rf_metrics,
 'rf_mean_regret_ci95':bootstrap_mean(df.regret),
 'rf_accuracy_ci95':bootstrap_mean(df.correct.astype(float)),
 'family_fixed_mean_regret':float(ff.regret.mean()),
 'global_fixed_mean_regret':float(gf.regret.mean()),
 'numba_mean_regret':float(df.numba_regret.mean()),
 'numpy_mean_regret':float(df.numpy_regret.mean()),
 'family_fixed_mean_regret_ci95':bootstrap_mean(ff.regret),
 'numba_mean_regret_ci95':bootstrap_mean(df.numba_regret),
 'case_level_rf_mean_regret':float(df.groupby(['heldout_rank','workload','size']).regret.mean().mean()),
 'case_level_rf_accuracy':float(df.groupby(['heldout_rank','workload','size']).correct.mean().mean()),
}
(RESULTS/'v07_predictions.json').write_text(json.dumps(fold_rows,indent=2))
(RESULTS/'v07_summary.json').write_text(json.dumps(summary,indent=2))
(RESULTS/'v07_family_fixed.json').write_text(json.dumps(family_fixed,indent=2))
(RESULTS/'v07_global_fixed.json').write_text(json.dumps(global_fixed,indent=2))
print(json.dumps(summary,indent=2))
