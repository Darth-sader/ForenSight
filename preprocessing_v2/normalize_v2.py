"""Partition-isolated feature normalization for ForenSight v2.

Fixes Leakage Risk LR-02:
In v1, normalization statistics were computed across concatenated arrays of all partitions.
FeatureNormalizerV2 strictly fits parameters on the training partition and transforms
validation and test partitions out-of-sample.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np


class FeatureNormalizerV2:
    """Standardizes or min-max normalizes features with strict partition isolation."""

    def __init__(self, method: str = "standard", eps: float = 1e-6):
        if method not in ("standard", "minmax", "robust"):
            raise ValueError(f"Unknown normalization method: {method}")
        self.method = method
        self.eps = eps
        self.fitted_ = False
        self.n_features_: int = 0
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizerV2":
        """Fit normalization parameters strictly on training partition matrix X."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")

        self.n_features_ = X.shape[1]

        if self.method == "standard":
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0) + self.eps
        elif self.method == "minmax":
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
        elif self.method == "robust":
            self.median_ = np.median(X, axis=0)
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            self.iqr_ = (q75 - q25) + self.eps

        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted parameters to X out-of-sample without modifying fitted statistics."""
        if not self.fitted_:
            raise RuntimeError("FeatureNormalizerV2 must be fitted before transforming data.")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Feature count mismatch: expected {self.n_features_}, got {X.shape[1]}")

        if self.method == "standard":
            assert self.mean_ is not None and self.std_ is not None
            out = (X - self.mean_) / self.std_
        elif self.method == "minmax":
            assert self.min_ is not None and self.max_ is not None
            denom = np.maximum(self.max_ - self.min_, self.eps)
            out = (X - self.min_) / denom
        elif self.method == "robust":
            assert self.median_ is not None and self.iqr_ is not None
            out = (X - self.median_) / self.iqr_

        # Clean any accidental NaNs or Infs from extreme out-of-bounds inputs
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureNormalizerV2":
        return joblib.load(path)
