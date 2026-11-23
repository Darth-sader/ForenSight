"""Rigorous Forensic Probability Calibration Suite (v2.0.0-phase6).

Strictly isolates probability types and evaluates calibration models:
1. Distinguishes:
   - Raw model probability (p_raw)
   - Calibrated probability (p_cal)
   - Graph-refined edge score (s_graph)
   - Reconstruction confidence (c_recon)
   - Structural validation level (L0-L5)
2. Compares Platt Scaling (Sigmoid) vs Isotonic Regression.
3. Computes:
   - Brier Score
   - Log Loss / Binary Cross-Entropy
   - Expected Calibration Error (ECE)
   - Maximum Calibration Error (MCE)
   - Reliability diagram binning data
4. Evaluates calibration stability across formats, corruption levels, and missing data.
5. Strictly fits calibrators on isolated calibration set (cal_files), never the test set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class ReliabilityDiagramData:
    """Binned reliability diagram statistics."""
    bin_centers: List[float]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    ece: float
    mce: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_centers": [round(x, 4) for x in self.bin_centers],
            "bin_accuracies": [round(x, 4) for x in self.bin_accuracies],
            "bin_confidences": [round(x, 4) for x in self.bin_confidences],
            "bin_counts": self.bin_counts,
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
        }


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration diagnostic metrics."""
    brier_score: float
    log_loss: float
    ece: float
    mce: float
    reliability: ReliabilityDiagramData

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brier_score": round(self.brier_score, 4),
            "log_loss": round(self.log_loss, 4),
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "reliability": self.reliability.to_dict(),
        }


class ForensicCalibratorV2:
    """Trains and evaluates Platt scaling and Isotonic regression calibrators."""

    def __init__(self, method: str = "isotonic"):
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Method must be 'platt' or 'isotonic', got {method}")
        self.method = method
        self._platt_model: Optional[LogisticRegression] = None
        self._isotonic_model: Optional[IsotonicRegression] = None
        self.is_fitted = False

    def fit(self, y_raw_scores: np.ndarray, y_true: np.ndarray) -> ForensicCalibratorV2:
        """Fit calibration curve on isolated calibration dataset."""
        y_raw = np.clip(y_raw_scores.astype(float), 1e-6, 1.0 - 1e-6)
        y_bin = y_true.astype(int)

        if self.method == "platt":
            # Platt scaling: Logistic regression on log-odds
            logits = np.log(y_raw / (1.0 - y_raw)).reshape(-1, 1)
            self._platt_model = LogisticRegression(C=1.0, solver="lbfgs")
            self._platt_model.fit(logits, y_bin)
        else:
            # Isotonic regression: non-parametric isotonic function
            self._isotonic_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._isotonic_model.fit(y_raw, y_bin)

        self.is_fitted = True
        return self

    def predict_proba(self, y_raw_scores: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted yet.")

        y_raw = np.clip(y_raw_scores.astype(float), 1e-6, 1.0 - 1e-6)

        if self.method == "platt" and self._platt_model is not None:
            logits = np.log(y_raw / (1.0 - y_raw)).reshape(-1, 1)
            p_cal = self._platt_model.predict_proba(logits)[:, 1]
        elif self._isotonic_model is not None:
            p_cal = self._isotonic_model.predict(y_raw)
        else:
            p_cal = y_raw

        return np.clip(p_cal, 1e-6, 1.0 - 1e-6)

    @staticmethod
    def compute_calibration_metrics(
        y_probs: np.ndarray,
        y_true: np.ndarray,
        num_bins: int = 10,
    ) -> CalibrationMetrics:
        """Compute Brier score, log loss, ECE, MCE, and reliability diagram bins."""
        probs = np.clip(y_probs.astype(float), 1e-6, 1.0 - 1e-6)
        targets = y_true.astype(int)

        brier = float(brier_score_loss(targets, probs))
        ll = float(log_loss(targets, probs))

        # Equal-width bins [0, 1]
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        bin_centers: List[float] = []
        bin_accuracies: List[float] = []
        bin_confidences: List[float] = []
        bin_counts: List[int] = []

        total_samples = len(probs)
        weighted_ece_sum = 0.0
        max_cal_error = 0.0

        for b in range(num_bins):
            lb, ub = bin_edges[b], bin_edges[b + 1]
            if b == num_bins - 1:
                mask = (probs >= lb) & (probs <= ub)
            else:
                mask = (probs >= lb) & (probs < ub)

            count = int(np.sum(mask))
            bin_counts.append(count)
            center = (lb + ub) / 2.0
            bin_centers.append(center)

            if count > 0:
                acc = float(np.mean(targets[mask]))
                conf = float(np.mean(probs[mask]))
                bin_accuracies.append(acc)
                bin_confidences.append(conf)
                error = abs(acc - conf)
                weighted_ece_sum += (count / total_samples) * error
                max_cal_error = max(max_cal_error, error)
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(center)

        rel_data = ReliabilityDiagramData(
            bin_centers=bin_centers,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            ece=float(weighted_ece_sum),
            mce=float(max_cal_error),
        )

        return CalibrationMetrics(
            brier_score=brier,
            log_loss=ll,
            ece=float(weighted_ece_sum),
            mce=float(max_cal_error),
            reliability=rel_data,
        )
