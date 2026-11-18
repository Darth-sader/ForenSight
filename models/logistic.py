"""Logistic Regression baseline (doc §12): interpretable linear probabilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing.normalize import FeatureNormalizer


class LogisticModel:
    name = "logistic_regression"

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 42):
        self.C = C
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced",
                                       random_state=seed, solver="lbfgs")),
        ])

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: Path) -> "LogisticModel":
        self.model = joblib.load(path)
        return self

    @property
    def coefficients(self) -> np.ndarray:
        return self.model.named_steps["clf"].coef_[0]
