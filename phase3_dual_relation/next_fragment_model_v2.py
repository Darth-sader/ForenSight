"""Dedicated Next-Fragment Transition Model (v2.0.0-phase3).

Predicts directed adjacency probability:
    P_next(i, j) = P(Fragment j immediately succeeds Fragment i).

Mathematical properties:
- Directed and asymmetric: P_next(i, j) != P_next(j, i) in general.
- Reverse-direction adjacency: P_prev(i, j) = P_next(j, i).
- Independent from same-file membership probability.
- Provides separate raw and Platt-calibrated probabilities.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class NextFragmentModelV2:
    """Predicts directed, asymmetric fragment adjacency probabilities."""

    name = "next_fragment_model_v2"

    def __init__(
        self,
        model_type: str = "random_forest",  # "random_forest" | "xgboost"
        n_estimators: int = 150,
        max_depth: Optional[int] = 16,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        self.model_type = model_type
        self.seed = seed
        self.calibrator_: Optional[CalibratedClassifierCV] = None

        if model_type == "random_forest":
            self.base_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=4,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=n_jobs,
            )
        elif model_type == "xgboost":
            self.base_model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth or 6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=4,
                eval_metric="logloss",
                tree_method="hist",
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train_next: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal_next: Optional[np.ndarray] = None,
    ) -> "NextFragmentModelV2":
        """Fit directed next-fragment classifier and calibrate on isolated calibration partition."""
        self.base_model.fit(X_train, y_train_next)

        # Fit Platt scaling (sigmoid calibrator) on disjoint calibration partition
        if X_cal is not None and y_cal_next is not None and len(X_cal) > 0:
            self.calibrator_ = CalibratedClassifierCV(
                estimator=self.base_model,
                method="sigmoid",
                cv="prefit",
            )
            self.calibrator_.fit(X_cal, y_cal_next)
        else:
            self.calibrator_ = None

        return self

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Compute uncalibrated next-fragment transition probabilities."""
        return self.base_model.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute calibrated probabilities if calibrator fitted, else raw."""
        if self.calibrator_ is not None:
            return self.calibrator_.predict_proba(X)[:, 1]
        return self.predict_proba_raw(X)

    def predict_proba_directed(
        self,
        X_forward: np.ndarray,
        X_reverse: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute both forward (i -> j) and reverse (j -> i) probabilities.

        Returns:
            (p_next_calibrated, p_prev_calibrated, p_next_raw, p_prev_raw)
        """
        p_next_raw = self.predict_proba_raw(X_forward)
        p_prev_raw = self.predict_proba_raw(X_reverse)

        if self.calibrator_ is not None:
            p_next_cal = self.calibrator_.predict_proba(X_forward)[:, 1]
            p_prev_cal = self.calibrator_.predict_proba(X_reverse)[:, 1]
        else:
            p_next_cal = p_next_raw
            p_prev_cal = p_prev_raw

        return p_next_cal, p_prev_cal, p_next_raw, p_prev_raw

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NextFragmentModelV2":
        return joblib.load(path)
