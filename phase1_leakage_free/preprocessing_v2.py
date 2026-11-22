"""Leakage-Free Preprocessing Pipeline (v2.0.0-leakage_free).

Guarantees:
1. Strict train-only fitting: all scalers, imputers, selectors, and dimensionality
   reduction components fit exclusively on the training partition.
2. Frozen transformation: validation, calibration, and test sets are transformed out-of-sample
   without modifying fitted parameters.
3. Zero corpus-wide statistics leakage (no global max_offset, global feature means, etc.).
4. Isolated artifact serialization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class LeakageFreePreprocessor:
    """Manages train-only fitting and out-of-sample application of preprocessing transformers."""

    def __init__(
        self,
        scaler_type: str = "standard",  # "standard", "minmax", "robust", "none"
        variance_threshold: Optional[float] = 1e-5,
        impute_strategy: str = "median",
    ):
        self.scaler_type = scaler_type
        self.variance_threshold = variance_threshold
        self.impute_strategy = impute_strategy

        self.fitted_: bool = False
        self.imputer_: Optional[SimpleImputer] = None
        self.variance_filter_: Optional[VarianceThreshold] = None
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None

        self.n_input_features_: int = 0
        self.n_output_features_: int = 0
        self.fitted_feature_names_: List[str] = []

    def fit(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "LeakageFreePreprocessor":
        """Fit all preprocessing parameters strictly on X_train."""
        X = np.asarray(X_train, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X.shape}")

        self.n_input_features_ = X.shape[1]
        self.fitted_feature_names_ = list(feature_names) if feature_names else [f"feat_{i}" for i in range(self.n_input_features_)]

        # 1. Imputer (train-only)
        self.imputer_ = SimpleImputer(strategy=self.impute_strategy)
        X_imputed = self.imputer_.fit_transform(X)

        # 2. Variance Threshold (train-only)
        if self.variance_threshold is not None:
            self.variance_filter_ = VarianceThreshold(threshold=self.variance_threshold)
            X_filtered = self.variance_filter_.fit_transform(X_imputed)
            # Update feature names mask
            support = self.variance_filter_.get_support()
            self.fitted_feature_names_ = [name for name, keep in zip(self.fitted_feature_names_, support) if keep]
        else:
            self.variance_filter_ = None
            X_filtered = X_imputed

        # 3. Scaler (train-only)
        if self.scaler_type == "standard":
            self.scaler_ = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler_ = MinMaxScaler()
        elif self.scaler_type == "robust":
            self.scaler_ = RobustScaler()
        elif self.scaler_type == "none":
            self.scaler_ = None
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

        if self.scaler_ is not None:
            self.scaler_.fit(X_filtered)

        self.n_output_features_ = X_filtered.shape[1]
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply frozen transformation out-of-sample without modifying fitted parameters."""
        if not self.fitted_:
            raise RuntimeError("LeakageFreePreprocessor must be fitted on training data before transform!")

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_input_features_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_input_features_}, got {X.shape[1]}"
            )

        # 1. Impute
        assert self.imputer_ is not None
        X_out = self.imputer_.transform(X)

        # 2. Variance Filter
        if self.variance_filter_ is not None:
            X_out = self.variance_filter_.transform(X_out)

        # 3. Scaler
        if self.scaler_ is not None:
            X_out = self.scaler_.transform(X_out)

        # Numeric hygiene
        X_out = np.nan_to_num(X_out, nan=0.0, posinf=1.0, neginf=-1.0)
        return X_out.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(X_train, feature_names=feature_names).transform(X_train)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LeakageFreePreprocessor":
        return joblib.load(path)
