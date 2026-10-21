from __future__ import annotations
import json, math, sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'; PLOTS=ROOT/'plots'; RES.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)
CASES=json.loads((RES/'v09_cases.json').read_text())
BACKENDS=('python','numpy','numba','numba_parallel')
SEEDS=[7,17,29,41,73]
WORKLOADS=sorted({c['workload'] for c in CASES})

NUMERIC=['size_log','elements_log','bytes_log','parallelism_log','arithmetic_intensity','dimensions','dtype_size',
         'window_log','assets_log','scenarios_log','paths_log','steps_log']

def feat(c, include_workload=True, groups='full'):
    s=c['spec']; md=s.get('metadata') or {}
    base={
      'size_log':float(np.log1p(s['size'])),'elements_log':float(np.log1p(s['elements'])),'bytes_log':float(np.log1p(s['bytes'])),
      'parallelism_log':float(np.log1p(s['parallelism'])),'arithmetic_intensity':float(s['arithmetic_intensity']),
      'dimensions':int(s['dimensions']),'dtype_size':int(s['dtype_size']),
      'window_log':float(np.log1p(md.get('window',0))),'assets_log':float(np.log1p(md.get('assets',0))),
      'scenarios_log':float(np.log1p(md.get('scenarios',0))),'paths_log':float(np.log1p(md.get('paths',0))),
      'steps_log':float(np.log1p(md.get('steps',0))) }
    if groups=='size_only': base={k:base[k] for k in ['size_log','elements_log','bytes_log']}
    elif groups=='structure': base={k:base[k] for k in ['dimensions','dtype_size','arithmetic_intensity','parallelism_log','window_log','assets_log','scenarios_log','paths_log','steps_log']}
    if include_workload: base['workload']=s['name']
    return base

def regret(c,b): return c['medians'][b]/c['medians'][c['oracle']]-1.0

def clf_fit(train, seed, include_workload=True, groups='full'):
    X=pd.DataFrame([feat(c,include_workload,groups) for c in train]); y=[c['oracle'] for c in train]
    cats=['workload'] if include_workload else []
    nums=[x for x in X.columns if x!='workload']
    steps=[]
    if cats: steps.append(('cat',OneHotEncoder(handle_unknown='ignore'),cats))
    steps.append(('num','passthrough',nums))
    pre=ColumnTransformer(steps)
    clf=RandomForestClassifier(n_estimators=300,random_state=seed,min_samples_leaf=1,class_weight='balanced_subsample',n_jobs=-1)
    return Pipeline([('pre',pre),('clf',clf)]).fit(X,y)

def clf_pred(model,c,include_workload=True,groups='full'):
    X=pd.DataFrame([feat(c,include_workload,groups)]); p=model.predict_proba(X)[0]; cls=list(model.named_steps['clf'].classes_)
    probs={k:float(v) for k,v in zip(cls,p)}; b=max(probs,key=probs.get)
    ordered=sorted(probs.items(),key=lambda kv:kv[1],reverse=True)
    margin=(ordered[0][1]-ordered[1][1]) if len(ordered)>1 else 1.0
    entropy=-sum(v*math.log(max(v,1e-12)) for v in probs.values())/math.log(len(BACKENDS))
    return b,probs,margin,entropy

def runtime_model_fit(train, seed, include_workload=True):
    rows=[feat(c,include_workload,'full') for c in train]; X=pd.DataFrame(rows)
    cats=['workload'] if include_workload else []; nums=[x for x in X.columns if x!='workload']
    steps=[]
    if cats: steps.append(('cat',OneHotEncoder(handle_unknown='ignore'),cats))
    steps.append(('num','passthrough',nums))
    pre=ColumnTransformer(steps)
    models={}
    for b in BACKENDS:
        reg=RandomForestRegressor(n_estimators=250,random_state=seed,min_samples_leaf=2,n_jobs=-1,bootstrap=True)
        models[b]=Pipeline([('pre',pre),('reg',reg)]).fit(X,np.log(np.array([c['medians'][b] for c in train])))
    return models

def runtime_pred(models,c,lam=.5,include_workload=True):
    X=pd.DataFrame([feat(c,include_workload,'full')]); out=[]
    for b,m in models.items():
        pre=m.named_steps['pre']; rf=m.named_steps['reg']; Xe=pre.transform(X)
        vals=np.vstack([t.predict(Xe) for t in rf.estimators_]).ravel(); mu=float(vals.mean()); sd=float(vals.std(ddof=1))
        mean=float(np.exp(mu+0.5*sd*sd)); q90=float(np.exp(mu+1.2816*sd)); unc=max(q90-mean,0.0); score=mean+lam*unc
        out.append((b,score,mean,unc))
    out.sort(key=lambda z:z[1]); return out[0][0],out

def bootstrap(vals, seed=20260915, n=5000):
    rng=np.random.default_rng(seed); a=np.asarray(vals,float); means=rng.choice(a,(n,len(a))).mean(axis=1)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def summarize(df):
    return {
      'n':int(len(df)),'accuracy':float(df.correct.mean()),'mean_regret':float(df.regret.mean()),
      'median_regret':float(df.regret.median()),'p95_regret':float(df.regret.quantile(.95)),'max_regret':float(df.regret.max()),
      'win_vs_numba':float((df.regret<df.numba_regret).mean()),'mean_numba_regret':float(df.numba_regret.mean()),
      'accuracy_ci95':bootstrap(df.correct.astype(float)),'mean_regret_ci95':bootstrap(df.regret)}

# Primary nested rank-holdout.
by_w={w:sorted([c for c in CASES if c['workload']==w],key=lambda z:z['size']) for w in WORKLOADS}
rows=[]
methods=[('rf_full',True,'full'),('rf_numeric',False,'full'),('rf_size_only',False,'size_only'),('rf_structure',False,'structure')]
for rank in range(5):
    train=[]; test=[]
    for w,fam in by_w.items():
        test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
    for seed in SEEDS:
        models={name:clf_fit(train,seed,inc,grp) for name,inc,grp in methods}
        rmods=runtime_model_fit(train,seed,include_workload=True)
        for c in test:
            for name,inc,grp in methods:
                pred,probs,margin,entropy=clf_pred(models[name],c,inc,grp)
                rows.append({'protocol':'rank_holdout','rank':rank,'seed':seed,'method':name,'workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':pred,'correct':pred==c['oracle'],'regret':regret(c,pred),'numba_regret':regret(c,'numba'),'margin':margin,'entropy':entropy,'probs':probs})
            pred,preds=runtime_pred(rmods,c,.5,True)
            rows.append({'protocol':'rank_holdout','rank':rank,'seed':seed,'method':'risk_regressor','workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':pred,'correct':pred==c['oracle'],'regret':regret(c,pred),'numba_regret':regret(c,'numba'),'margin':None,'entropy':None,'risk_scores':[(b,s,m,u) for b,s,m,u in preds]})

primary=pd.DataFrame(rows)
summary_primary={m:summarize(g) for m,g in primary.groupby('method')}

# Baselines: global/family fixed, selected strictly from training portion.
base_rows=[]
for rank in range(5):
    train=[]; test=[]
    for w,fam in by_w.items():
        test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
    global_backend=Counter(c['oracle'] for c in train).most_common(1)[0][0]
    for c in test:
        fam_train=[x for x in train if x['workload']==c['workload']]
        fam_backend=Counter(x['oracle'] for x in fam_train).most_common(1)[0][0]
        base_rows += [
          {'method':'global_fixed','rank':rank,'workload':c['workload'],'size':c['size'],'backend':global_backend,'regret':regret(c,global_backend)},
          {'method':'family_fixed','rank':rank,'workload':c['workload'],'size':c['size'],'backend':fam_backend,'regret':regret(c,fam_backend)},
          {'method':'oracle','rank':rank,'workload':c['workload'],'size':c['size'],'backend':c['oracle'],'regret':0.0}]
baseline_df=pd.DataFrame(base_rows)
summary_baselines={m:{'mean_regret':float(g.regret.mean()),'median_regret':float(g.regret.median()),'p95_regret':float(g.regret.quantile(.95))} for m,g in baseline_df.groupby('method')}

# Cross-family generalization: one entire family is unseen; numeric-only methods only.
cf=[]
for held in WORKLOADS:
    train=[c for c in CASES if c['workload']!=held]; test=[c for c in CASES if c['workload']==held]
    for seed in SEEDS:
        clf=clf_fit(train,seed,False,'full')
        rm=runtime_model_fit(train,seed,False)
        for c in test:
            pred,probs,margin,entropy=clf_pred(clf,c,False,'full')
            cf.append({'heldout_family':held,'seed':seed,'method':'rf_numeric','workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':pred,'correct':pred==c['oracle'],'regret':regret(c,pred)})
            pred2,_=runtime_pred(rm,c,.5,False)
            cf.append({'heldout_family':held,'seed':seed,'method':'risk_regressor_numeric','workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':pred2,'correct':pred2==c['oracle'],'regret':regret(c,pred2)})
cfdf=pd.DataFrame(cf)
summary_cross={m:summarize(g) for m,g in cfdf.groupby('method')}

# Targeted ablation: for the risk regressor, remove risk penalty (lambda=0), compare lambda=.5.
ab=[]
for rank in range(5):
    train=[]; test=[]
    for w,fam in by_w.items():
        test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
    for seed in SEEDS:
        rm=runtime_model_fit(train,seed,True)
        for c in test:
            for lam in [0.0,0.5,1.0,2.0]:
                pred,_=runtime_pred(rm,c,lam,True)
                ab.append({'rank':rank,'seed':seed,'lambda':lam,'workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':pred,'correct':pred==c['oracle'],'regret':regret(c,pred)})
abdf=pd.DataFrame(ab)
summary_lambda=abdf.groupby('lambda').agg(accuracy=('correct','mean'),mean_regret=('regret','mean'),median_regret=('regret','median'),p95_regret=('regret',lambda x: x.quantile(.95))).reset_index().to_dict(orient='records')

# Output.
(RES/'v09_primary_predictions.json').write_text(primary.to_json(orient='records',indent=2))
(RES/'v09_baselines.json').write_text(baseline_df.to_json(orient='records',indent=2))
(RES/'v09_cross_family.json').write_text(cfdf.to_json(orient='records',indent=2))
(RES/'v09_lambda_ablation.json').write_text(abdf.to_json(orient='records',indent=2))
summary={'version':'0.9','dataset':'20 freshly measured cases: 4 finance workloads x 5 sizes; 3 steady-state repeats after 1 warmup; fixed seed 73','protocol':'rank-holdout across all families; 5 seeds; plus leave-one-family-out cross-family generalization','primary':summary_primary,'baselines':summary_baselines,'cross_family':summary_cross,'lambda_ablation':summary_lambda,'measurement_note':'All timings in v09_cases.json are measured on the same runtime host during this session; no synthetic timings were used.'}
(RES/'v09_summary.json').write_text(json.dumps(summary,indent=2))

# Plots.
import matplotlib.pyplot as plt
morder=['global_fixed','family_fixed','rf_full','rf_numeric','rf_size_only','rf_structure','risk_regressor']
vals=[]; labels=[]
for m in morder:
    src=summary_baselines[m]['mean_regret'] if m in summary_baselines else summary_primary[m]['mean_regret']
    labels.append(m); vals.append(src*100)
plt.figure(figsize=(10,5)); plt.bar(labels,vals); plt.ylabel('Mean regret (%)'); plt.xticks(rotation=30,ha='right'); plt.title('AQX v0.9 primary comparison'); plt.tight_layout(); plt.savefig(PLOTS/'v09_primary_regret.png',dpi=160); plt.close()

cross_labels=list(summary_cross); cross_vals=[summary_cross[x]['mean_regret']*100 for x in cross_labels]
plt.figure(figsize=(8,4.5)); plt.bar(cross_labels,cross_vals); plt.ylabel('Mean regret (%)'); plt.title('Leave-one-family-out generalization'); plt.tight_layout(); plt.savefig(PLOTS/'v09_cross_family.png',dpi=160); plt.close()

plt.figure(figsize=(7,4.5)); d=abdf.groupby('lambda')['regret'].mean()*100; plt.plot(d.index,d.values,marker='o'); plt.xlabel('Risk penalty λ'); plt.ylabel('Mean regret (%)'); plt.title('Risk penalty ablation'); plt.tight_layout(); plt.savefig(PLOTS/'v09_lambda_ablation.png',dpi=160); plt.close()

print(json.dumps(summary,indent=2))
