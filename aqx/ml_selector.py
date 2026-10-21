from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

BACKENDS = ("python", "numpy", "numba", "numba_parallel")

@dataclass
class Prediction:
    selected: str
    probabilities: dict[str, float]
    margin: float
    entropy: float
    abstained: bool
    reason: str
    def to_dict(self): return asdict(self)

def feature_row(spec) -> dict[str, Any]:
    md = spec.metadata or {}
    return {
        "workload": spec.name,
        "size_log": float(np.log1p(spec.size)),
        "elements_log": float(np.log1p(spec.elements)),
        "bytes_log": float(np.log1p(spec.bytes)),
        "parallelism_log": float(np.log1p(spec.parallelism)),
        "arithmetic_intensity": float(spec.arithmetic_intensity),
        "dimensions": int(spec.dimensions),
        "dtype_size": int(spec.dtype_size),
        "window_log": float(np.log1p(md.get("window", 0))),
        "assets_log": float(np.log1p(md.get("assets", 0))),
        "scenarios_log": float(np.log1p(md.get("scenarios", 0))),
        "paths_log": float(np.log1p(md.get("paths", 0))),
        "steps_log": float(np.log1p(md.get("steps", 0))),
    }

class BackendClassifier:
    """Probability model for which backend will be fastest."""
    NUMERIC = ["size_log","elements_log","bytes_log","parallelism_log","arithmetic_intensity","dimensions","dtype_size",
               "window_log","assets_log","scenarios_log","paths_log","steps_log"]
    CAT = ["workload"]
    def __init__(self, n_estimators=300, random_state=17, min_leaf=2):
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), self.CAT), ("num", "passthrough", self.NUMERIC)])
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, min_samples_leaf=min_leaf,
                                     class_weight="balanced_subsample", n_jobs=-1)
        self.pipeline = Pipeline([("pre", pre), ("clf", clf)])
        self.fitted = False
        self.classes_: list[str] = []
        self.train_log_loss: float | None = None
    def fit(self, records: list[dict]) -> None:
        X = pd.DataFrame([feature_row_from_record(r) for r in records])
        y = [r["oracle"] for r in records]
        self.pipeline.fit(X, y)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        p = self.pipeline.predict_proba(X)
        self.train_log_loss = None if len(self.classes_) < 2 else float(log_loss(y, p, labels=self.classes_))
        self.fitted = True
    def predict(self, spec, threshold=0.0, entropy_threshold=1.0) -> Prediction:
        if not self.fitted: raise RuntimeError("model not fitted")
        X = pd.DataFrame([feature_row(spec)])
        probs_arr = self.pipeline.predict_proba(X)[0]
        probs = {c: float(p) for c,p in zip(self.classes_, probs_arr)}
        ranked = sorted(probs, key=probs.get, reverse=True)
        best = ranked[0]
        second_p = probs[ranked[1]] if len(ranked) > 1 else 0.0
        margin = probs[best] - second_p
        entropy = float(-sum(p*np.log(max(p,1e-12)) for p in probs_arr))
        abstain = margin < threshold or entropy > entropy_threshold
        reason = "confident" if not abstain else ("low_margin" if margin < threshold else "high_entropy")
        return Prediction(best, probs, margin, entropy, abstain, reason)

def feature_row_from_record(r: dict) -> dict[str, Any]:
    spec = r["spec"]
    class Obj: pass
    o = Obj(); [setattr(o,k,v) for k,v in spec.items()]
    return feature_row(o)
