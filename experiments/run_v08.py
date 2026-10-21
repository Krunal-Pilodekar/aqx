from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aqx.risk_selector import RuntimeRiskModel,BACKENDS
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; PLOTS=ROOT/'plots'; RES.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)
CASES=json.loads((RES/'source_v05_cases.json').read_text())
SEEDS=[7,17,29,41,73]
LAMBDAS=[0.0,0.5,1.0,2.0]
by_w={w:sorted([c for c in CASES if c['workload']==w],key=lambda z:z['size']) for w in sorted({c['workload'] for c in CASES})}

def regret(c,b): return c['medians'][b]/c['medians'][c['oracle']]-1.0

def outer_rows(lam,seed):
    rows=[]
    for rank in range(4):
        train=[]; test=[]
        for w,fam in by_w.items():
            test.append(fam[rank]); train.extend([x for i,x in enumerate(fam) if i!=rank])
        model=RuntimeRiskModel(n_estimators=50,random_state=seed,min_samples_leaf=2).fit(train)
        for c in test:
            pred,preds=model.predict(c,lambda_risk=lam)
            rows.append({'rank':rank,'seed':seed,'workload':c['workload'],'size':c['size'],'oracle':c['oracle'],'predicted':pred,
                         'correct':pred==c['oracle'],'regret':regret(c,pred),'numba_regret':regret(c,'numba'),
                         'numpy_regret':regret(c,'numpy'),'predictions':[p.to_dict() for p in preds],
                         'oracle_time_s':c['medians'][c['oracle']], 'selected_time_s':c['medians'][pred]})
    return rows

def tune_lambda():
    # Inner-style validation using only 3 size ranks, evaluate next rank within the training portion.
    vals=[]
    for lam in LAMBDAS:
        for seed in [17,29]:
            for held_rank in [1,2]:
                train=[]; val=[]
                for w,fam in by_w.items():
                    for i,x in enumerate(fam):
                        if i==held_rank: val.append(x)
                        else: train.append(x)
                model=RuntimeRiskModel(n_estimators=50,random_state=seed,min_samples_leaf=2).fit(train)
                for c in val:
                    p,_=model.predict(c,lambda_risk=lam)
                    vals.append({'lambda':lam,'seed':seed,'rank':held_rank,'regret':regret(c,p),'predicted':p,'correct':p==c['oracle']})
    df=pd.DataFrame(vals)
    agg=df.groupby('lambda').agg(mean_regret=('regret','mean'),accuracy=('correct','mean')).reset_index()
    # Primary objective: mean regret; tie-break accuracy.
    best=agg.sort_values(['mean_regret','accuracy'],ascending=[True,False]).iloc[0]
    return float(best['lambda']),agg,df

best_lam,tune_df,tune_rows=tune_lambda()
tune_records=tune_rows.to_dict(orient='records') if isinstance(tune_rows,pd.DataFrame) else tune_rows
all_rows=[]
for seed in SEEDS: all_rows.extend(outer_rows(best_lam,seed))
df=pd.DataFrame(all_rows)

def bootstrap(vals,n=10000):
    a=np.asarray(vals,float); rng=np.random.default_rng(2026); means=[]
    for _ in range(n): means.append(rng.choice(a,len(a),replace=True).mean())
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
summary={'version':'0.8','best_lambda':best_lam,'tuning':tune_df.to_dict(orient='records'),
         'n_cases':len(CASES),'n_predictions':len(df),'seeds':SEEDS,
         'protocol':'nested rank-holdout: lambda tuned only on inner ranks 1-2; final evaluation uses outer leave-one-size-rank-out',
         'risk_model':{
             'accuracy':float(df.correct.mean()),'mean_regret':float(df.regret.mean()),'median_regret':float(df.regret.median()),
             'p95_regret':float(df.regret.quantile(.95)),'max_regret':float(df.regret.max()),
             'mean_numba_regret':float(df.numba_regret.mean()),'win_rate_vs_numba':float((df.regret<df.numba_regret).mean()),
             'accuracy_ci95':bootstrap(df.correct.astype(float)),'mean_regret_ci95':bootstrap(df.regret)
         }}
(RES/'v08_predictions.json').write_text(json.dumps(all_rows,indent=2))
(RES/'v08_inner_tuning.json').write_text(json.dumps(tune_records,indent=2))
(RES/'v08_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
