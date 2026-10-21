from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; PLOTS=ROOT/'plots'
CASES=json.loads((RES/'v09_cases.json').read_text())
BACKENDS=('python','numpy','numba','numba_parallel'); SEEDS=[7,29,73]
WORKLOADS=sorted({c['workload'] for c in CASES})

def feat(c, mode='full'):
    s=c['spec']; md=s.get('metadata') or {}
    d={'size_log':np.log1p(s['size']),'elements_log':np.log1p(s['elements']),'bytes_log':np.log1p(s['bytes']),
       'parallelism_log':np.log1p(s['parallelism']),'arithmetic_intensity':s['arithmetic_intensity'],'dimensions':s['dimensions'],'dtype_size':s['dtype_size'],
       'window_log':np.log1p(md.get('window',0)),'assets_log':np.log1p(md.get('assets',0)),'scenarios_log':np.log1p(md.get('scenarios',0)),
       'paths_log':np.log1p(md.get('paths',0)),'steps_log':np.log1p(md.get('steps',0))}
    if mode=='size': d={k:d[k] for k in ['size_log','elements_log','bytes_log']}
    elif mode=='structure': d={k:d[k] for k in ['parallelism_log','arithmetic_intensity','dimensions','dtype_size','window_log','assets_log','scenarios_log','paths_log','steps_log']}
    if mode=='full_workload': d['workload']=s['name']
    return d

def regret(c,b): return c['medians'][b]/c['medians'][c['oracle']]-1

def clf(train, seed, mode):
    X=pd.DataFrame([feat(c,mode) for c in train]); y=[c['oracle'] for c in train]
    cats=['workload'] if 'workload' in X.columns else []; nums=[x for x in X.columns if x!='workload']
    tr=[]
    if cats: tr.append(('cat',OneHotEncoder(handle_unknown='ignore'),cats))
    tr.append(('num',StandardScaler(),nums))
    pre=ColumnTransformer(tr)
    # Logistic regression is a deliberately strong, low-variance baseline; RF retained as nonlinear selector.
    if mode=='rf':
        real_mode='full_workload'; X=pd.DataFrame([feat(c,real_mode) for c in train]); cats=['workload']; nums=[x for x in X.columns if x!='workload']
        pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cats),('num','passthrough',nums)])
        m=RandomForestClassifier(n_estimators=100,random_state=seed,min_samples_leaf=1,n_jobs=-1,class_weight='balanced_subsample')
    else:
        m=LogisticRegression(max_iter=1000,random_state=seed)
    return Pipeline([('pre',pre),('model',m)]).fit(X,y)

def pred(m,c,mode):
    X=pd.DataFrame([feat(c, 'full_workload' if mode=='rf' else mode)]); p=m.predict_proba(X)[0]; cls=list(m.named_steps['model'].classes_); probs=dict(zip(cls,p)); b=max(probs,key=probs.get); return b,probs

def runtime_models(train,seed):
    X=pd.DataFrame([feat(c,'full') for c in train]); models={}
    for b in BACKENDS:
        m=RandomForestRegressor(n_estimators=80,random_state=seed,min_samples_leaf=2,n_jobs=-1)
        models[b]=(X,m.fit(X,np.log([c['medians'][b] for c in train])))
    return models

def rpred(models,c,lam):
    Xc=pd.DataFrame([feat(c,'full')]); out=[]
    for b,(X,m) in models.items():
        vals=np.vstack([t.predict(Xc) for t in m.estimators_]).ravel(); mu=vals.mean(); sd=vals.std(ddof=1)
        mean=np.exp(mu+.5*sd*sd); q90=np.exp(mu+1.2816*sd); unc=q90-mean; out.append((b,mean+lam*unc))
    return min(out,key=lambda z:z[1])[0]

def boot(a,n=3000):
    a=np.asarray(a); rng=np.random.default_rng(12345); return [float(np.quantile(rng.choice(a,(n,len(a))).mean(1),.025)),float(np.quantile(rng.choice(a,(n,len(a))).mean(1),.975))]

def summ(df):
    return {'n':int(len(df)),'accuracy':float(df.correct.mean()),'mean_regret':float(df.regret.mean()),'median_regret':float(df.regret.median()),'p95_regret':float(df.regret.quantile(.95)),'max_regret':float(df.regret.max()),'mean_numba_regret':float(df.numba_regret.mean()),'win_vs_numba':float((df.regret<df.numba_regret).mean()),'accuracy_ci95':boot(df.correct.astype(float)),'mean_regret_ci95':boot(df.regret)}

# Rank holdout; methods: RF, numeric LR, size-only LR, structure-only LR, risk regressor.
by_w={w:sorted([c for c in CASES if c['workload']==w],key=lambda c:c['size']) for w in WORKLOADS}
rows=[]
for rank in range(5):
  train=[]; test=[]
  for w,fam in by_w.items(): test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
  for seed in SEEDS:
    models={m:clf(train,seed,m) for m in ['rf','full','size','structure']}
    rms=runtime_models(train,seed)
    for c in test:
      for m,mm in models.items():
        mode='full_workload' if m=='rf' else m
        p,_=pred(mm,c,m); rows.append({'protocol':'rank_holdout','rank':rank,'seed':seed,'method':('rf_full' if m=='rf' else 'lr_'+m),'workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':p,'correct':p==c['oracle'],'regret':regret(c,p),'numba_regret':regret(c,'numba')})
      p=rpred(rms,c,.5); rows.append({'protocol':'rank_holdout','rank':rank,'seed':seed,'method':'risk_regressor','workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':p,'correct':p==c['oracle'],'regret':regret(c,p),'numba_regret':regret(c,'numba')})
primary=pd.DataFrame(rows)

# Fixed baselines.
base=[]
for rank in range(5):
  train=[]; test=[]
  for w,fam in by_w.items(): test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
  gb=Counter(c['oracle'] for c in train).most_common(1)[0][0]
  for c in test:
    fb=Counter(x['oracle'] for x in train if x['workload']==c['workload']).most_common(1)[0][0]
    base += [{'method':'global_fixed','rank':rank,'workload':c['workload'],'size':c['size'],'backend':gb,'regret':regret(c,gb)}, {'method':'family_fixed','rank':rank,'workload':c['workload'],'size':c['size'],'backend':fb,'regret':regret(c,fb)}]
base_df=pd.DataFrame(base)

# Cross-family: numeric-only LR, a deliberately difficult generalization test.
cf=[]
for held in WORKLOADS:
  train=[c for c in CASES if c['workload']!=held]; test=[c for c in CASES if c['workload']==held]
  for seed in SEEDS:
    m=clf(train,seed,'full'); rm=runtime_models(train,seed)
    for c in test:
      p,_=pred(m,c,'full'); cf.append({'held':held,'seed':seed,'method':'lr_numeric','correct':p==c['oracle'],'regret':regret(c,p),'numba_regret':regret(c,'numba')})
      p=rpred(rm,c,.5); cf.append({'held':held,'seed':seed,'method':'risk_regressor_numeric','correct':p==c['oracle'],'regret':regret(c,p),'numba_regret':regret(c,'numba')})
cfdf=pd.DataFrame(cf)

# Risk lambda ablation on rank-holdout using fixed .0/.5/1/2 values.
lam=[]
for rank in range(5):
  train=[]; test=[]
  for w,fam in by_w.items(): test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
  rms=runtime_models(train,17)
  for c in test:
    for L in [0.,.5,1.,2.]:
      p=rpred(rms,c,L); lam.append({'lambda':L,'regret':regret(c,p),'correct':p==c['oracle']})
lamdf=pd.DataFrame(lam)

# Save.
summary={'version':'0.9','dataset':'20 freshly measured cases; 4 finance workload families x 5 sizes; 3 steady-state repeats after warmup; fixed seed 73','primary':{m:summ(g) for m,g in primary.groupby('method')},'baselines':{m:{'mean_regret':float(g.regret.mean()),'median_regret':float(g.regret.median()),'p95_regret':float(g.regret.quantile(.95))} for m,g in base_df.groupby('method')},'cross_family':{m:summ(g) for m,g in cfdf.groupby('method')},'lambda_ablation':lamdf.groupby('lambda').agg(accuracy=('correct','mean'),mean_regret=('regret','mean'),p95_regret=('regret',lambda x:x.quantile(.95))).reset_index().to_dict('records'),'citation_context':['ML-based compiler autotuning is a mature research area; AQX is positioned as domain-specific adaptive backend selection rather than generic compiler autotuning.']}
(RES/'v09_primary_predictions.json').write_text(primary.to_json(orient='records',indent=2)); (RES/'v09_baselines.json').write_text(base_df.to_json(orient='records',indent=2)); (RES/'v09_cross_family.json').write_text(cfdf.to_json(orient='records',indent=2)); (RES/'v09_lambda_ablation.json').write_text(lamdf.to_json(orient='records',indent=2)); (RES/'v09_summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib.pyplot as plt
labels=list(summary['baselines'])+list(summary['primary']); vals=[summary['baselines'][x]['mean_regret']*100 for x in summary['baselines']]+[summary['primary'][x]['mean_regret']*100 for x in summary['primary']]
plt.figure(figsize=(10,5)); plt.bar(labels,vals); plt.ylabel('Mean regret (%)'); plt.title('AQX v0.9 comparison'); plt.xticks(rotation=30,ha='right'); plt.tight_layout(); plt.savefig(PLOTS/'v09_primary_regret.png',dpi=160); plt.close()
labels=list(summary['cross_family']); vals=[summary['cross_family'][x]['mean_regret']*100 for x in labels]; plt.figure(figsize=(8,4.5)); plt.bar(labels,vals); plt.ylabel('Mean regret (%)'); plt.title('Cross-family generalization'); plt.tight_layout(); plt.savefig(PLOTS/'v09_cross_family.png',dpi=160); plt.close()
labels=[str(r['lambda']) for r in summary['lambda_ablation']]; vals=[r['mean_regret']*100 for r in summary['lambda_ablation']]; plt.figure(figsize=(7,4.5)); plt.plot(labels,vals,marker='o'); plt.xlabel('Risk penalty λ'); plt.ylabel('Mean regret (%)'); plt.title('Risk penalty ablation'); plt.tight_layout(); plt.savefig(PLOTS/'v09_lambda_ablation.png',dpi=160); plt.close()
print(json.dumps(summary,indent=2))
