from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; PLOTS=ROOT/'plots'
CASES=json.loads((RES/'v09_cases.json').read_text())
BACKENDS=('python','numpy','numba','numba_parallel'); SEEDS=[7,29,73]; WORKLOADS=sorted({c['workload'] for c in CASES})

def feat(c,mode):
 s=c['spec']; md=s.get('metadata') or {}
 d={'size_log':np.log1p(s['size']),'elements_log':np.log1p(s['elements']),'bytes_log':np.log1p(s['bytes']),
    'parallelism_log':np.log1p(s['parallelism']),'arithmetic_intensity':s['arithmetic_intensity'],'dimensions':s['dimensions'],'dtype_size':s['dtype_size'],
    'window_log':np.log1p(md.get('window',0)),'assets_log':np.log1p(md.get('assets',0)),'scenarios_log':np.log1p(md.get('scenarios',0)),'paths_log':np.log1p(md.get('paths',0)),'steps_log':np.log1p(md.get('steps',0))}
 if mode=='size': return {k:d[k] for k in ['size_log','elements_log','bytes_log']}
 if mode=='full_workload': d['workload']=s['name']
 return d

def regret(c,b): return c['medians'][b]/c['medians'][c['oracle']]-1

def fit(train,seed,mode):
 X=pd.DataFrame([feat(c,mode) for c in train]); y=[c['oracle'] for c in train]
 if len(set(y))<2:
  return ('constant', y[0])
 cats=['workload'] if 'workload' in X else []; nums=[x for x in X.columns if x!='workload']
 tr=[]
 if cats: tr.append(('cat',OneHotEncoder(handle_unknown='ignore'),cats))
 tr.append(('num',StandardScaler(),nums))
 return Pipeline([('pre',ColumnTransformer(tr)),('clf',LogisticRegression(max_iter=300,random_state=seed))]).fit(X,y)

def pred(model,c,mode):
 if isinstance(model,tuple): return model[1]
 X=pd.DataFrame([feat(c,mode)]); return model.predict(X)[0]

def boot(a,n=5000):
 a=np.asarray(a,float); rng=np.random.default_rng(20260915); means=rng.choice(a,(n,len(a))).mean(1); return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def summarize(df):
 return {'n':int(len(df)),'accuracy':float(df.correct.mean()),'mean_regret':float(df.regret.mean()),'median_regret':float(df.regret.median()),'p95_regret':float(df.regret.quantile(.95)),'max_regret':float(df.regret.max()),'accuracy_ci95':boot(df.correct),'mean_regret_ci95':boot(df.regret),'mean_numba_regret':float(df.numba_regret.mean()),'win_vs_numba':float((df.regret<df.numba_regret).mean())}

# 5-fold leave-one-size-rank-out.
by={w:sorted([c for c in CASES if c['workload']==w],key=lambda c:c['size']) for w in WORKLOADS}
rows=[]
for rank in range(5):
 train=[]; test=[]
 for w,fam in by.items(): test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
 for seed in SEEDS:
  for mode,name in [('full_workload','LR+family'),('full','LR+numeric'),('size','LR+size')]:
   m=fit(train,seed,mode)
   for c in test:
    p=pred(m,c,mode)
    rows.append({'protocol':'rank_holdout','rank':rank,'seed':seed,'method':name,'workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':p,'correct':p==c['oracle'],'regret':regret(c,p),'numba_regret':regret(c,'numba')})
primary=pd.DataFrame(rows)

# Fixed baselines chosen from training only.
base=[]
for rank in range(5):
 train=[]; test=[]
 for w,fam in by.items(): test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
 gb=Counter(x['oracle'] for x in train).most_common(1)[0][0]
 for c in test:
  fb=Counter(x['oracle'] for x in train if x['workload']==c['workload']).most_common(1)[0][0]
  base.extend([{'method':'global_fixed','regret':regret(c,gb),'workload':c['workload'],'size':c['size']},{'method':'family_fixed','regret':regret(c,fb),'workload':c['workload'],'size':c['size']}])
basedf=pd.DataFrame(base)

# Cross-family generalization.
cf=[]
for held in WORKLOADS:
 train=[c for c in CASES if c['workload']!=held]; test=[c for c in CASES if c['workload']==held]
 for seed in SEEDS:
  m=fit(train,seed,'full')
  for c in test:
   p=pred(m,c,'full'); cf.append({'heldout_family':held,'seed':seed,'correct':p==c['oracle'],'regret':regret(c,p),'numba_regret':regret(c,'numba'),'oracle':c['oracle'],'predicted':p,'workload':c['workload'],'size':c['size']})
cfdf=pd.DataFrame(cf)

# Class-weight / model complexity ablation using regularization strengths on rank holdout.
ab=[]
for C in [0.1,1,10]:
 for rank in range(5):
  train=[]; test=[]
  for w,fam in by.items(): test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
  for seed in [17]:
   X=pd.DataFrame([feat(c,'full') for c in train]); y=[c['oracle'] for c in train]
   m=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(max_iter=300,C=C,random_state=seed))]).fit(X,y)
   for c in test:
    p=m.predict(pd.DataFrame([feat(c,'full')]))[0]
    ab.append({'C':C,'regret':regret(c,p),'correct':p==c['oracle']})
abdf=pd.DataFrame(ab)

summary={'version':'0.9','dataset':'20 freshly measured cases; 4 workloads x 5 sizes; 3 steady-state repeats after warmup; fixed RNG seed 73','primary':{m:summarize(g) for m,g in primary.groupby('method')},'baselines':{m:{'mean_regret':float(g.regret.mean()),'median_regret':float(g.regret.median()),'p95_regret':float(g.regret.quantile(.95))} for m,g in basedf.groupby('method')},'cross_family':summarize(cfdf),'ablation_C':abdf.groupby('C').agg(accuracy=('correct','mean'),mean_regret=('regret','mean'),p95_regret=('regret',lambda x:x.quantile(.95))).reset_index().to_dict('records'),'literature_note':'The literature on ML-based compiler autotuning is mature, so AQX is evaluated as a domain-specific backend-selection problem rather than claiming generic autotuning novelty.'}
(RES/'v09_primary_predictions.json').write_text(primary.to_json(orient='records',indent=2)); (RES/'v09_baselines.json').write_text(basedf.to_json(orient='records',indent=2)); (RES/'v09_cross_family.json').write_text(cfdf.to_json(orient='records',indent=2)); (RES/'v09_ablation.json').write_text(abdf.to_json(orient='records',indent=2)); (RES/'v09_summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib.pyplot as plt
labels=list(summary['baselines'])+list(summary['primary']); vals=[summary['baselines'][m]['mean_regret']*100 for m in summary['baselines']]+[summary['primary'][m]['mean_regret']*100 for m in summary['primary']]
plt.figure(figsize=(10,5)); plt.bar(labels,vals); plt.ylabel('Mean regret (%)'); plt.title('AQX v0.9 primary comparison'); plt.xticks(rotation=30,ha='right'); plt.tight_layout(); plt.savefig(PLOTS/'v09_primary_regret.png',dpi=160); plt.close()
plt.figure(figsize=(7,4.5)); vals=[summary['cross_family']['accuracy']*100, summary['cross_family']['mean_regret']*100]; plt.bar(['Accuracy %','Mean regret %'],vals); plt.title('Cross-family numeric model'); plt.tight_layout(); plt.savefig(PLOTS/'v09_cross_family.png',dpi=160); plt.close()
print(json.dumps(summary,indent=2))
