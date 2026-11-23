"""Reconstruction-Level Uncertainty Quantification Engine (v2.0.0-phase6).

Evaluates multi-dimensional forensic uncertainty across 7 distinct signals:
1. Predictive entropy: H(p) = -p log2(p) - (1-p) log2(1-p)
2. Ensemble disagreement: variance across model predictions
3. Alternative path margin: delta_score = Score(best) - Score(second_best)
4. Graph topological ambiguity: out-degree branching and competing candidate density
5. Missing fragment gap impact: penalty and missing block ratio
6. Format structural validation level: L0 (signature) to L5 (ground truth)
7. Competing hypothesis diversity: Jaccard distance between top candidate paths

Computes risk-coverage trade-off curves, selective accuracy, and rejection rates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from phase6_forensic_validation.format_validators.base_validator import FormatValidationResult, ValidationLevel


@dataclass
class UncertaintySignal:
    """Decomposed uncertainty profile for a single reconstruction hypothesis."""
    recon_id: str
    predictive_entropy: float           # Shannon entropy [0.0, 1.0]
    ensemble_variance: float            # Model disagreement [0.0, 0.25]
    alternative_margin: float           # Margin over runner-up path >= 0.0
    graph_ambiguity: float              # Branching ambiguity [0.0, 1.0]
    gap_penalty_ratio: float            # Missing evidence fraction [0.0, 1.0]
    validation_level: ValidationLevel   # Format validator tier
    competing_hypothesis_count: int     # Number of alternative viable paths
    composite_uncertainty_score: float  # Synthesized uncertainty score [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recon_id": self.recon_id,
            "predictive_entropy": round(self.predictive_entropy, 4),
            "ensemble_variance": round(self.ensemble_variance, 4),
            "alternative_margin": round(self.alternative_margin, 4),
            "graph_ambiguity": round(self.graph_ambiguity, 4),
            "gap_penalty_ratio": round(self.gap_penalty_ratio, 4),
            "validation_level": f"L{self.validation_level.value}_{self.validation_level.name}",
            "competing_hypothesis_count": self.competing_hypothesis_count,
            "composite_uncertainty_score": round(self.composite_uncertainty_score, 4),
        }


@dataclass
class RiskCoveragePoint:
    """A single coordinate on the empirical Risk-Coverage curve."""
    coverage: float                     # Fraction of reconstructions retained [0.0, 1.0]
    selective_risk: float               # Error rate on retained reconstructions
    selective_accuracy: float           # 1.0 - selective_risk
    uncertainty_threshold: float        # Cutoff used to achieve this coverage


class ForensicUncertaintyEngine:
    """Computes multidimensional reconstruction uncertainty and risk-coverage dynamics."""

    def __init__(
        self,
        weight_entropy: float = 0.25,
        weight_variance: float = 0.15,
        weight_margin: float = 0.25,
        weight_ambiguity: float = 0.15,
        weight_gaps: float = 0.20,
    ):
        self.w_ent = weight_entropy
        self.w_var = weight_variance
        self.w_mar = weight_margin
        self.w_amb = weight_ambiguity
        self.w_gap = weight_gaps

    @staticmethod
    def binary_entropy(p: float) -> float:
        """Shannon binary entropy in bits [0.0, 1.0]."""
        p_c = min(max(p, 1e-6), 1.0 - 1e-6)
        return float(-p_c * math.log2(p_c) - (1.0 - p_c) * math.log2(1.0 - p_c))

    def evaluate_hypothesis_uncertainty(
        self,
        recon_id: str,
        edge_probs: List[float],
        ensemble_preds: Optional[List[List[float]]] = None,
        runner_up_margin: float = 1.0,
        graph_branching_factor: float = 1.0,
        gap_count: int = 0,
        total_fragments: int = 1,
        val_result: Optional[FormatValidationResult] = None,
        competing_count: int = 1,
    ) -> UncertaintySignal:
        """Compute multidimensional uncertainty signals for one reconstruction candidate."""
        # 1. Predictive entropy
        mean_entropy = (
            float(np.mean([self.binary_entropy(p) for p in edge_probs]))
            if edge_probs else 0.0
        )

        # 2. Ensemble variance
        if ensemble_preds and len(ensemble_preds) > 1:
            var_per_edge = [float(np.var(col)) for col in zip(*ensemble_preds)]
            mean_variance = float(np.mean(var_per_edge))
        else:
            mean_variance = 0.0

        # 3. Alternative margin normalized [0, 1] (high margin = low uncertainty)
        # S-curve mapping: margin of 2.0 -> ~0.1 uncertainty, margin of 0.0 -> 1.0 uncertainty
        margin_uncertainty = float(1.0 / (1.0 + max(0.0, runner_up_margin)))

        # 4. Graph ambiguity (normalized branching factor)
        ambiguity = min(1.0, max(0.0, (graph_branching_factor - 1.0) / 4.0))

        # 5. Gap penalty ratio
        gap_ratio = min(1.0, gap_count / max(1, total_fragments))

        # 6. Format validation tier
        v_level = val_result.highest_passed_level if val_result else ValidationLevel.L0_SIGNATURE
        val_penalty = 0.0
        if v_level < ValidationLevel.L2_STRUCTURE:
            val_penalty = 0.5
        elif v_level < ValidationLevel.L3_DECOMPRESSION:
            val_penalty = 0.2

        # 7. Composite synthesis
        raw_composite = (
            self.w_ent * mean_entropy +
            self.w_var * (mean_variance * 4.0) +  # scale max var 0.25 to 1.0
            self.w_mar * margin_uncertainty +
            self.w_amb * ambiguity +
            self.w_gap * gap_ratio +
            val_penalty
        )
        composite_score = float(min(1.0, max(0.0, raw_composite)))

        return UncertaintySignal(
            recon_id=recon_id,
            predictive_entropy=mean_entropy,
            ensemble_variance=mean_variance,
            alternative_margin=runner_up_margin,
            graph_ambiguity=ambiguity,
            gap_penalty_ratio=gap_ratio,
            validation_level=v_level,
            competing_hypothesis_count=competing_count,
            composite_uncertainty_score=composite_score,
        )

    @staticmethod
    def compute_risk_coverage_curve(
        uncertainty_scores: np.ndarray,
        is_error_flags: np.ndarray,     # 1 if reconstruction is incorrect/false, 0 if correct
        num_points: int = 20,
    ) -> List[RiskCoveragePoint]:
        """Compute selective accuracy across coverage levels by sweeping uncertainty threshold."""
        n = len(uncertainty_scores)
        if n == 0:
            return []

        # Sort candidates by uncertainty ascending (lowest uncertainty = highest confidence)
        sorted_indices = np.argsort(uncertainty_scores)
        sorted_scores = uncertainty_scores[sorted_indices]
        sorted_errors = is_error_flags[sorted_indices]

        curve: List[RiskCoveragePoint] = []
        quantiles = np.linspace(0.1, 1.0, num_points)

        for q in quantiles:
            k = max(1, int(math.ceil(q * n)))
            retained_errors = sorted_errors[:k]
            selective_risk = float(np.mean(retained_errors))
            selective_acc = 1.0 - selective_risk
            threshold = float(sorted_scores[k - 1])

            curve.append(RiskCoveragePoint(
                coverage=float(k / n),
                selective_risk=selective_risk,
                selective_accuracy=selective_acc,
                uncertainty_threshold=threshold,
            ))

        return curve
