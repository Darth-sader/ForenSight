"""Baseline supervised models (doc §12): Logistic Regression, Random Forest,
XGBoost gradient boosting, plus the deep-learning components (doc §13-14).

All models expose: ``fit(X, y, X_val, y_val)``, ``predict_proba(X)`` and
``save/load`` for artifact persistence.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


class ForensicModel(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "ForensicModel": ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)
