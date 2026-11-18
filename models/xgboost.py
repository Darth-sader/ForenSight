"""XGBoost gradient boosting (doc §12): primary tabular ML candidate."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from xgboost import XGBClassifier


class XGBoostModel:
    name = "xgboost"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.1,
                 max_depth: int = 6, subsample: float = 0.9,
                 colsample_bytree: float = 0.8, seed: int = 42, n_jobs: int = 4):
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           max_depth=max_depth, subsample=subsample,
                           colsample_bytree=colsample_bytree)
        self.model = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            subsample=subsample, colsample_bytree=colsample_bytree,
            eval_metric="logloss", random_state=seed, n_jobs=n_jobs,
            tree_method="hist", verbosity=0)

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is not None and len(X_val):
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: Path) -> "XGBoostModel":
        self.model = joblib.load(path)
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
