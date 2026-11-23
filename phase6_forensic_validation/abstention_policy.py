"""Cost-Sensitive Forensic Abstention and Decision Policy (v2.0.0-phase6).

Classifies reconstruction candidates into four formal evidentiary categories:
- ACCEPTED: Candidate satisfies structural parsing, decoding, and high-confidence bounds.
- PARTIAL: Candidate contains verified sub-chains but has unresolved gaps or missing blocks.
- UNCERTAIN: Multiple competing hypotheses exist or evidence margins are insufficient.
- REJECTED: Candidate violates structural requirements (fails L1) or lacks evidentiary support.

Threshold selection is determined using validation data under a formal forensic cost model:
    Risk = C_false_accept * P(False Accept) + C_reject * P(Reject) + C_partial * P(Partial)
where C_false_accept >> C_reject to guard courtroom/evidentiary integrity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phase6_forensic_validation.format_validators.base_validator import FormatValidationResult, ValidationLevel
from phase6_forensic_validation.uncertainty_engine import UncertaintySignal


class ReconstructionOutcome(str, Enum):
    """Formal evidentiary disposition for a reconstruction candidate."""
    ACCEPTED = "ACCEPTED"
    PARTIAL = "PARTIAL"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


@dataclass
class PolicyCostModel:
    """Investigative cost model reflecting forensic evidence risk."""
    cost_false_accept: float = 10.0   # Severe cost for introducing fabricated or merged files
    cost_false_reject: float = 1.0    # Cost of missing recoverable evidence
    cost_partial_accept: float = 0.5  # Modest cost for accepting partial with caveat disclosures
    cost_uncertain: float = 0.75      # Cost of flagging for manual expert review


@dataclass
class DecisionVerdict:
    """Forensic decision record for a single reconstruction hypothesis."""
    recon_id: str
    outcome: ReconstructionOutcome
    achieved_validation_level: ValidationLevel
    composite_uncertainty: float
    decision_rationale: str
    is_forensically_usable: bool       # True for ACCEPTED or qualified PARTIAL
    requires_human_inspection: bool    # True for UNCERTAIN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recon_id": self.recon_id,
            "outcome": self.outcome.value,
            "achieved_validation_level": f"L{self.achieved_validation_level.value}_{self.achieved_validation_level.name}",
            "composite_uncertainty": round(self.composite_uncertainty, 4),
            "decision_rationale": self.decision_rationale,
            "is_forensically_usable": self.is_forensically_usable,
            "requires_human_inspection": self.requires_human_inspection,
        }


class CostSensitiveAbstentionPolicy:
    """Applies evidence-based thresholds derived from forensic cost optimization."""

    def __init__(
        self,
        cost_model: Optional[PolicyCostModel] = None,
        tau_accept: float = 0.35,      # Max composite uncertainty permitted for ACCEPTED
        tau_partial: float = 0.65,     # Max composite uncertainty permitted for PARTIAL
        min_margin_accept: float = 0.5 # Min alternative path margin required for ACCEPTED
    ):
        self.costs = cost_model or PolicyCostModel()
        self.tau_accept = tau_accept
        self.tau_partial = tau_partial
        self.min_margin_accept = min_margin_accept

    def calibrate_thresholds_on_validation(
        self,
        val_uncertainties: List[UncertaintySignal],
        val_error_flags: List[int],    # 1 if invalid/corrupt, 0 if valid
        val_gap_flags: List[int],      # 1 if has missing blocks, 0 if complete
    ) -> Dict[str, float]:
        """Grid search over validation data to find threshold minimizing expected risk."""
        best_risk = float("inf")
        best_tau_accept = 0.35
        best_tau_partial = 0.65

        candidate_taus = np.linspace(0.15, 0.60, 10)

        for t_acc in candidate_taus:
            for t_part in np.linspace(t_acc + 0.1, 0.85, 8):
                total_risk = 0.0
                for unc, err, gap in zip(val_uncertainties, val_error_flags, val_gap_flags):
                    score = unc.composite_uncertainty_score
                    if score <= t_acc and unc.validation_level >= ValidationLevel.L2_STRUCTURE:
                        outcome = ReconstructionOutcome.ACCEPTED
                    elif score <= t_part and unc.validation_level >= ValidationLevel.L1_LAYOUT:
                        outcome = ReconstructionOutcome.PARTIAL
                    elif unc.competing_hypothesis_count > 1 and unc.alternative_margin < 0.2:
                        outcome = ReconstructionOutcome.UNCERTAIN
                    else:
                        outcome = ReconstructionOutcome.REJECTED

                    # Apply cost model
                    if outcome == ReconstructionOutcome.ACCEPTED:
                        if err == 1:
                            total_risk += self.costs.cost_false_accept
                    elif outcome == ReconstructionOutcome.PARTIAL:
                        if err == 1:
                            total_risk += self.costs.cost_false_accept
                        else:
                            total_risk += self.costs.cost_partial_accept
                    elif outcome == ReconstructionOutcome.UNCERTAIN:
                        total_risk += self.costs.cost_uncertain
                    elif outcome == ReconstructionOutcome.REJECTED:
                        if err == 0:
                            total_risk += self.costs.cost_false_reject

                if total_risk < best_risk:
                    best_risk = total_risk
                    best_tau_accept = float(t_acc)
                    best_tau_partial = float(t_part)

        self.tau_accept = best_tau_accept
        self.tau_partial = best_tau_partial

        return {
            "optimal_tau_accept": round(best_tau_accept, 4),
            "optimal_tau_partial": round(best_tau_partial, 4),
            "minimized_expected_risk": round(best_risk, 4),
        }

    def evaluate_candidate(
        self,
        signal: UncertaintySignal,
        val_result: FormatValidationResult,
    ) -> DecisionVerdict:
        """Classify candidate into ACCEPTED, PARTIAL, UNCERTAIN, or REJECTED."""
        score = signal.composite_uncertainty_score
        v_level = val_result.highest_passed_level

        # 1. Structural violation check: failing L1 layout is an immediate rejection
        if v_level < ValidationLevel.L1_LAYOUT or not val_result.is_structurally_valid:
            if v_level == ValidationLevel.L0_SIGNATURE and signal.gap_penalty_ratio > 0.5:
                return DecisionVerdict(
                    recon_id=signal.recon_id,
                    outcome=ReconstructionOutcome.REJECTED,
                    achieved_validation_level=v_level,
                    composite_uncertainty=score,
                    decision_rationale=f"Failed basic structural layout verification ({val_result.errors}).",
                    is_forensically_usable=False,
                    requires_human_inspection=False,
                )

        # 2. Ambiguity check: high competing hypotheses with razor-thin margin
        if signal.competing_hypothesis_count > 1 and signal.alternative_margin < self.min_margin_accept:
            return DecisionVerdict(
                recon_id=signal.recon_id,
                outcome=ReconstructionOutcome.UNCERTAIN,
                achieved_validation_level=v_level,
                composite_uncertainty=score,
                decision_rationale=(
                    f"Ambiguous alternative paths detected ({signal.competing_hypothesis_count} hypotheses, "
                    f"margin={signal.alternative_margin:.3f} < {self.min_margin_accept})."
                ),
                is_forensically_usable=False,
                requires_human_inspection=True,
            )

        # 3. Acceptance criteria: low uncertainty and successful structural parsing/decoding
        if score <= self.tau_accept and v_level >= ValidationLevel.L2_STRUCTURE and signal.gap_penalty_ratio == 0.0:
            return DecisionVerdict(
                recon_id=signal.recon_id,
                outcome=ReconstructionOutcome.ACCEPTED,
                achieved_validation_level=v_level,
                composite_uncertainty=score,
                decision_rationale="Meets full evidence criteria: structural parsing passed, zero unresolved gaps, low uncertainty.",
                is_forensically_usable=True,
                requires_human_inspection=False,
            )

        # 4. Partial criteria: verified structure with declared gaps or moderate uncertainty
        if score <= self.tau_partial and (v_level >= ValidationLevel.L1_LAYOUT or signal.gap_penalty_ratio > 0.0):
            return DecisionVerdict(
                recon_id=signal.recon_id,
                outcome=ReconstructionOutcome.PARTIAL,
                achieved_validation_level=v_level,
                composite_uncertainty=score,
                decision_rationale=(
                    f"Qualified partial reconstruction: structure verified at level {v_level.name}, "
                    f"gap ratio={signal.gap_penalty_ratio:.2f}."
                ),
                is_forensically_usable=True,
                requires_human_inspection=False,
            )

        # 5. Fallback: High uncertainty or unresolved errors
        return DecisionVerdict(
            recon_id=signal.recon_id,
            outcome=ReconstructionOutcome.REJECTED,
            achieved_validation_level=v_level,
            composite_uncertainty=score,
            decision_rationale=f"Uncertainty ({score:.3f}) exceeds threshold ({self.tau_partial}) or validation incomplete.",
            is_forensically_usable=False,
            requires_human_inspection=False,
        )
