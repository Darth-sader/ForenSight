"""Random Forest ensemble baseline (doc §12): nonlinear baseline + feature
importance for explainability (doc §37)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    name = "random_forest"

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = 24,
                 min_samples_split: int = 4, seed: int = 42, n_jobs: int = -1):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           min_samples_split=min_samples_split)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, class_weight="balanced_subsample",
            random_state=seed, n_jobs=n_jobs)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: Path) -> "RandomForestModel":
        self.model = joblib.load(path)
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
