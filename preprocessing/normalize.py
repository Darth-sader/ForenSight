"""Data preprocessing (doc §8): byte/numerical normalisation, missing-value
handling, feature scaling, dimensionality reduction and duplicate detection."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def normalize_bytes(data_matrix: np.ndarray) -> np.ndarray:
    """Byte normalisation: raw bytes -> [0, 1] floats."""
    return data_matrix.astype(np.float32) / 255.0


class FeatureNormalizer:
    """Missing-value handling + scaling, fitted on train data only."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        X_imp = self.imputer.fit_transform(X)
        self.scaler.fit(X_imp)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        return self.scaler.transform(X_imp).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PCAReducer:
    """PCA used as preprocessing/analysis technique (doc §15)."""

    def __init__(self, n_components: int = 32):
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit(self, X: np.ndarray) -> "PCAReducer":
        self.pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(X).astype(np.float32)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_


def detect_duplicates(data_matrix: np.ndarray) -> np.ndarray:
    """Boolean mask of duplicate byte blocks (exact content repeats)."""
    seen: Dict[bytes, bool] = {}
    dup = np.zeros(data_matrix.shape[0], dtype=bool)
    for i in range(data_matrix.shape[0]):
        key = data_matrix[i].tobytes()
        if key in seen:
            dup[i] = True
        else:
            seen[key] = True
    return dup


def train_val_test_group_split(n_groups: int, val_frac: float = 0.15,
                               test_frac: float = 0.15, seed: int = 42
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group split by *file* so no file leaks across splits (doc §26)."""
    rng = np.random.RandomState(seed)
    groups = rng.permutation(n_groups)
    n_val = max(1, int(n_groups * val_frac))
    n_test = max(1, int(n_groups * test_frac))
    return groups[n_val + n_test:], groups[:n_val], groups[n_val:n_val + n_test]
