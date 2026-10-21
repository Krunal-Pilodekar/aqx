from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BACKENDS=("python","numpy","numba","numba_parallel")

NUMERIC=["size_log","elements_log","bytes_log","parallelism_log","arithmetic_intensity","dimensions","dtype_size",
         "window_log","assets_log","scenarios_log","paths_log","steps_log"]
CAT=["workload"]

@dataclass
class RiskPrediction:
    backend:str
    predicted_time_s:float
    uncertainty_s:float
    risk_score:float
    q90_time_s:float

    def to_dict(self): return asdict(self)

def feature_row_from_case(c:dict)->dict[str,Any]:
    spec=c["spec"]
    md=spec.get("metadata") or {}
    size=float(spec["size"]); elements=float(spec["elements"]); bytes_=float(spec["bytes"])
    return {
        "workload":spec["name"],
        "size_log":float(np.log1p(size)),"elements_log":float(np.log1p(elements)),"bytes_log":float(np.log1p(bytes_)),
        "parallelism_log":float(np.log1p(spec["parallelism"])),"arithmetic_intensity":float(spec["arithmetic_intensity"]),
        "dimensions":int(spec["dimensions"]),"dtype_size":int(spec["dtype_size"]),
        "window_log":float(np.log1p(md.get("window",0))),"assets_log":float(np.log1p(md.get("assets",0))),
        "scenarios_log":float(np.log1p(md.get("scenarios",0))),"paths_log":float(np.log1p(md.get("paths",0))),
        "steps_log":float(np.log1p(md.get("steps",0))),
    }

class RuntimeRiskModel:
    """Per-backend runtime models. Trees provide an empirical prediction distribution."""
    def __init__(self,n_estimators=500,random_state=17,min_samples_leaf=2):
        self.models={}; self.n_estimators=n_estimators; self.random_state=random_state; self.min_samples_leaf=min_samples_leaf
    def _pipeline(self):
        pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),CAT),("num","passthrough",NUMERIC)])
        reg=RandomForestRegressor(n_estimators=self.n_estimators,random_state=self.random_state,
                                  min_samples_leaf=self.min_samples_leaf,n_jobs=-1,bootstrap=True)
        return Pipeline([("pre",pre),("reg",reg)])
    def fit(self,cases:list[dict]):
        X=pd.DataFrame([feature_row_from_case(c) for c in cases])
        for b in BACKENDS:
            self.models[b]=self._pipeline(); self.models[b].fit(X,np.log(np.array([c["medians"][b] for c in cases])))
        return self
    def predict(self,c:dict,lambda_risk:float=1.0,z:float=1.2816)->tuple[str,list[RiskPrediction]]:
        X=pd.DataFrame([feature_row_from_case(c)]); preds=[]
        for b,m in self.models.items():
            rf=m.named_steps["reg"]
            Xenc=m.named_steps["pre"].transform(X)
            tree_log=np.vstack([t.predict(Xenc) for t in rf.estimators_]).ravel()
            mean_log=float(tree_log.mean()); sd_log=float(tree_log.std(ddof=1)) if len(tree_log)>1 else 0.0
            # Lognormal approximation for positive runtime.
            mean_t=float(np.exp(mean_log + 0.5*sd_log**2)); q90=float(np.exp(mean_log + z*sd_log))
            unc=float(q90-mean_t)
            score=float(mean_t + lambda_risk*unc)
            preds.append(RiskPrediction(b,mean_t,unc,score,q90))
        preds.sort(key=lambda x:x.risk_score)
        return preds[0].backend,preds
