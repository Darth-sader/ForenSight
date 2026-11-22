"""Standardized Dual-Relation Inference Engine (v2.0.0-phase3).

Provides a strict, decoupled inference interface that computes and separates:
1. Symmetric same-file membership probability: P_same(i, j)
2. Directed next-fragment transition probability: P_next(i, j)
3. Reverse-direction transition probability: P_prev(i, j) = P_next(j, i)

Explicitly prevents and detects identical-probability copying or reuse.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from phase3_dual_relation.multitask_model import MultiTaskDualModel
from phase3_dual_relation.next_fragment_model_v2 import NextFragmentModelV2
from phase3_dual_relation.same_file_model_v2 import SameFileModelV2


class ProbabilityReuseError(ValueError):
    """Raised when same-file and next-fragment probabilities are illegally reused or conflated."""
    pass


@dataclass
class DualRelationScores:
    """Standardized score container for a fragment pair (F_i, F_j)."""
    p_same: float          # Calibrated symmetric same-file membership probability
    p_next: float          # Calibrated forward directed adjacency probability P(j follows i)
    p_prev: float          # Calibrated reverse directed adjacency probability P(i follows j)
    p_same_raw: float      # Uncalibrated same-file probability
    p_next_raw: float      # Uncalibrated forward adjacency probability
    confidence: float      # Uncertainty-calibrated confidence score
    model_version: str     # Provenance model identifier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_same": self.p_same,
            "p_next": self.p_next,
            "p_prev": self.p_prev,
            "p_same_raw": self.p_same_raw,
            "p_next_raw": self.p_next_raw,
            "confidence": self.confidence,
            "model_version": self.model_version,
        }


class DualRelationInferenceEngine:
    """Evaluates fragment pairs using independent models or unified multi-task architecture."""

    def __init__(
        self,
        same_model: Optional[Union[SameFileModelV2, Any]] = None,
        next_model: Optional[Union[NextFragmentModelV2, Any]] = None,
        multitask_model: Optional[MultiTaskDualModel] = None,
        version: str = "2.0.0-phase3",
    ):
        self.version = version
        self.multitask_model = multitask_model
        self.same_model = same_model
        self.next_model = next_model

        if multitask_model is None and (same_model is None or next_model is None):
            raise ValueError("Must provide either multitask_model or both same_model and next_model.")

    def compute_pair_scores(
        self,
        X_fwd: np.ndarray,
        X_rev: Optional[np.ndarray] = None,
        validate_non_identity: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Compute dual relationship scores for a batch of fragment pairs.

        Args:
            X_fwd: (N, D) pairwise features for (F_i, F_j)
            X_rev: Optional (N, D) pairwise features for (F_j, F_i)
            validate_non_identity: Enforces P_same != P_next across the batch

        Returns:
            Dictionary containing:
                'p_same': (N,) array of symmetric same-file probabilities
                'p_next': (N,) array of directed forward probabilities
                'p_prev': (N,) array of directed reverse probabilities
                'p_same_raw': (N,) uncalibrated same-file probabilities
                'p_next_raw': (N,) uncalibrated forward probabilities
                'confidence': (N,) uncertainty confidence scores
        """
        if X_rev is None:
            # If reverse features are not provided, default to symmetric approximation
            X_rev = X_fwd

        if self.multitask_model is not None:
            p_same, p_next, p_prev = self.multitask_model.predict_proba(X_fwd, X_rev)
            p_same_raw, p_next_raw, _ = self.multitask_model.predict_raw(X_fwd, X_rev)
        else:
            assert self.same_model is not None and self.next_model is not None
            p_same, p_same_raw = self.same_model.predict_proba_symmetric(X_fwd, X_rev)
            p_next, p_prev, p_next_raw, _ = self.next_model.predict_proba_directed(X_fwd, X_rev)

        p_same = np.asarray(p_same, dtype=np.float64)
        p_next = np.asarray(p_next, dtype=np.float64)
        p_prev = np.asarray(p_prev, dtype=np.float64)
        p_same_raw = np.asarray(p_same_raw, dtype=np.float64)
        p_next_raw = np.asarray(p_next_raw, dtype=np.float64)

        if validate_non_identity and len(p_same) > 5:
            # Check for identical probability copying vulnerability
            diff = np.abs(p_same - p_next)
            if np.all(diff < 1e-7):
                raise ProbabilityReuseError(
                    "P_same and P_next are numerically identical across all pairs! "
                    "Detected illegal probability reuse / copying."
                )

        # Calibrated joint confidence: high when both relationship signals agree or have high certainty
        confidence = 0.5 * (np.abs(p_same - 0.5) * 2.0 + np.abs(p_next - 0.5) * 2.0)

        return {
            "p_same": p_same,
            "p_next": p_next,
            "p_prev": p_prev,
            "p_same_raw": p_same_raw,
            "p_next_raw": p_next_raw,
            "confidence": confidence,
        }

    def score_single_pair(
        self,
        x_fwd: np.ndarray,
        x_rev: Optional[np.ndarray] = None,
    ) -> DualRelationScores:
        """Compute dual relationship scores for an individual pair."""
        batch_fwd = x_fwd.reshape(1, -1)
        batch_rev = x_rev.reshape(1, -1) if x_rev is not None else None
        res = self.compute_pair_scores(batch_fwd, batch_rev, validate_non_identity=False)
        return DualRelationScores(
            p_same=float(res["p_same"][0]),
            p_next=float(res["p_next"][0]),
            p_prev=float(res["p_prev"][0]),
            p_same_raw=float(res["p_same_raw"][0]),
            p_next_raw=float(res["p_next_raw"][0]),
            confidence=float(res["confidence"][0]),
            model_version=self.version,
        )
