"""Confidence calibration (doc §23): Platt scaling, isotonic regression,
Brier score, Expected Calibration Error and reliability curves."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt scaling: logistic regression on raw model scores."""

    name = "platt"

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr.fit(raw.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(raw.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibration (non-parametric)."""

    name = "isotonic"

    def __init__(self, out_of_bounds: str = "clip"):
        self.iso = IsotonicRegression(out_of_bounds=out_of_bounds, y_min=0.0, y_max=1.0)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(raw, y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self.iso.predict(raw)


def fit_best_calibrator(raw: np.ndarray, y: np.ndarray) -> object:
    """Fit both calibrators, return the one with the lower Brier score."""
    best = None
    best_brier = float("inf")
    for cal in (PlattCalibrator(), IsotonicCalibrator()):
        cal.fit(raw, y)
        b = brier_score(cal.transform(raw), y)
        if b < best_brier:
            best, best_brier = cal, b
    return best


def brier_score(proba: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((proba - y) ** 2))


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass-bin ECE (doc §23)."""
    order = np.argsort(proba)
    bins = np.array_split(order, n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        conf = proba[b].mean()
        acc = y[b].mean()
        ece += len(b) / len(proba) * abs(conf - acc)
    return float(ece)


def reliability_curve(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, list]:
    order = np.argsort(proba)
    bins = np.array_split(order, n_bins)
    conf, acc, size = [], [], []
    for b in bins:
        if len(b) == 0:
            continue
        conf.append(float(proba[b].mean()))
        acc.append(float(y[b].mean()))
        size.append(int(len(b)))
    return {"confidence": conf, "accuracy": acc, "bin_size": size}
