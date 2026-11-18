"""Anomaly detection (doc §17): Isolation Forest and Local Outlier Factor
(plus One-Class SVM where useful) to flag statistically unusual, structurally
inconsistent or possibly misassigned fragments."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def _to_score(raw: np.ndarray) -> np.ndarray:
    """Map sklearn's {-1 (outlier) .. 1 (inlier)} decision values to 0..1
    anomaly scores (higher = more anomalous)."""
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        return np.full_like(raw, 0.5, dtype=np.float64)
    return 1.0 - (raw - lo) / (hi - lo)


class IsolationForestDetector:
    name = "isolation_forest"

    def __init__(self, contamination: float = 0.05, seed: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=seed)

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        return _to_score(self.model.decision_function(X))

    def labels(self, X: np.ndarray) -> np.ndarray:
        return (self.model.predict(X) == -1).astype(int)


class LOFDetector:
    name = "local_outlier_factor"

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model: Optional[LocalOutlierFactor] = None

    def fit(self, X: np.ndarray) -> "LOFDetector":
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                        contamination=self.contamination, novelty=True)
        self.model.fit(X)
        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return _to_score(self.model.decision_function(X))

    def labels(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return (self.model.predict(X) == -1).astype(int)


class OneClassSVMDetector:
    name = "one_class_svm"

    def __init__(self, nu: float = 0.05, kernel: str = "rbf"):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        self.model.fit(X)
        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        return _to_score(self.model.decision_function(X))

    def labels(self, X: np.ndarray) -> np.ndarray:
        return (self.model.predict(X) == -1).astype(int)


def anomaly_status(score: float, threshold: float = 0.7) -> str:
    return "SUSPICIOUS" if score >= threshold else "NORMAL"


def combine_scores(scores_list, weights=None) -> np.ndarray:
    """Weighted combination of multiple detectors' scores."""
    if weights is None:
        weights = [1.0 / len(scores_list)] * len(scores_list)
    total = np.zeros_like(scores_list[0], dtype=np.float64)
    for s, w in zip(scores_list, weights):
        total += w * s
    return np.clip(total, 0.0, 1.0)
