"""Dedicated Same-File Membership Model (v2.0.0-phase3).

Predicts P_same(i, j) = P(F_i and F_j belong to the same original file).
Enforces mathematical symmetry: P_same(i, j) == P_same(j, i).
Provides separate raw and calibrated probabilities.
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


class SameFileModelV2:
    """Predicts symmetric same-file fragment membership probabilities."""

    name = "same_file_model_v2"

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
        y_train_same: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal_same: Optional[np.ndarray] = None,
    ) -> "SameFileModelV2":
        """Fit same-file classifier and calibrate on isolated calibration partition."""
        self.base_model.fit(X_train, y_train_same)

        # Fit isotonic / sigmoid calibrator on calibration partition if provided
        if X_cal is not None and y_cal_same is not None and len(X_cal) > 0:
            self.calibrator_ = CalibratedClassifierCV(
                estimator=self.base_model,
                method="sigmoid",  # Platt scaling
                cv="prefit",
            )
            self.calibrator_.fit(X_cal, y_cal_same)
        else:
            self.calibrator_ = None

        return self

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Compute uncalibrated same-file probabilities."""
        return self.base_model.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute calibrated probabilities if calibrator fitted, else raw."""
        if self.calibrator_ is not None:
            return self.calibrator_.predict_proba(X)[:, 1]
        return self.predict_proba_raw(X)

    def predict_proba_symmetric(self, X_forward: np.ndarray, X_reverse: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enforce exact mathematical symmetry: P_same(i, j) == P_same(j, i).
        
        Returns:
            (calibrated_symmetric_prob, raw_symmetric_prob)
        """
        p_raw_fwd = self.predict_proba_raw(X_forward)
        p_raw_rev = self.predict_proba_raw(X_reverse)
        p_raw_sym = 0.5 * (p_raw_fwd + p_raw_rev)

        if self.calibrator_ is not None:
            p_cal_fwd = self.calibrator_.predict_proba(X_forward)[:, 1]
            p_cal_rev = self.calibrator_.predict_proba(X_reverse)[:, 1]
            p_cal_sym = 0.5 * (p_cal_fwd + p_cal_rev)
        else:
            p_cal_sym = p_raw_sym

        return p_cal_sym, p_raw_sym

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SameFileModelV2":
        return joblib.load(path)
