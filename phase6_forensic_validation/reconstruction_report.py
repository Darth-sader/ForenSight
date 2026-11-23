"""Investigator-Readable Forensic Reconstruction Audit Report (v2.0.0-phase6).

Generates court-admissible, investigator-readable forensic reconstruction reports:
- Case, acquisition, and chain-of-custody headers
- Evidentiary disposition (ACCEPTED, PARTIAL, UNCERTAIN, REJECTED)
- Verified structural validation tiers (L0 to L5)
- Fragment sequence table with physical offsets, lengths, and cryptographic hashes
- Separate confidence reporting (raw vs calibrated probabilities vs entropy)
- Unresolved gap, missing block, and corruption disclosures
- Rejected alternative paths and runner-up margins
- Cryptographic provenance signature for reproducible verification
- Mandatory legal and scientific disclosure (ML scores do not constitute proof of authenticity)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from phase6_forensic_validation.abstention_policy import DecisionVerdict, ReconstructionOutcome
from phase6_forensic_validation.format_validators.base_validator import FormatValidationResult
from phase6_forensic_validation.provenance import EvidenceProvenanceRecord
from phase6_forensic_validation.uncertainty_engine import UncertaintySignal


class ForensicReconstructionReportGenerator:
    """Produces standardized Markdown and plain-text forensic reconstruction audit reports."""

    @staticmethod
    def generate_markdown_report(
        provenance: EvidenceProvenanceRecord,
        uncertainty: Optional[UncertaintySignal] = None,
        rejected_hypotheses: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Compile comprehensive, investigator-readable forensic reconstruction report."""
        v = provenance.verdict or {}
        val = provenance.validation or {}

        outcome_str = v.get("outcome", "UNCERTAIN")
        val_level = val.get("highest_passed_level", "L0_SIGNATURE")
        rationale = v.get("decision_rationale", "No rationale documented.")

        lines: List[str] = [
            "# FORENSIC FILE RECONSTRUCTION AUDIT REPORT",
            "**Case / Evidence Identifier:** " + provenance.disk_image_id,
            f"**Reconstruction ID:** `{provenance.recon_id}` | **Software Version:** `{provenance.software_version}`",
            f"**Generated (UTC):** {provenance.timestamp_utc} | **Acquisition Source:** {provenance.acquisition_source}",
            "",
            "---",
            "",
            "## 1. EVIDENTIARY VERDICT & SUMMARY",
            f"- **Disposition Outcome:** **`{outcome_str}`**",
            f"- **Structural Validation Tier:** `{val_level}`",
            f"- **Claimed Format:** `{provenance.claimed_format}`",
            f"- **Reconstructed Size:** `{provenance.reconstructed_size_bytes:,} bytes` ({len(provenance.fragments)} fragments)",
            f"- **SHA-256 Digest:** `{provenance.reconstructed_sha256}`",
            f"- **MD5 Digest:** `{provenance.reconstructed_md5}`",
            f"- **Decision Rationale:** {rationale}",
            f"- **Court / Investigative Admissibility:** " + ("Qualified for investigative use" if v.get("is_forensically_usable") else "NOT qualified (abstained or rejected)"),
            "",
            "---",
            "",
            "## 2. COMPONENT FRAGMENT CHAIN OF CUSTODY",
            "| Position | Fragment ID | Physical Offset | Length (Bytes) | Allocation Status | SHA-256 Hash (First 16 chars) |",
            "| :---: | :--- | :---: | :---: | :---: | :--- |",
        ]

        for idx, frag in enumerate(provenance.fragments):
            h_short = frag.sha256_hash[:16] + "..."
            lines.append(
                f"| {idx:02d} | `{frag.fragment_id}` | `{frag.observed_offset:,}` | "
                f"`{frag.fragment_length:,}` | `{frag.allocation_status}` | `{h_short}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. UNCERTAINTY & TRANSITION CONFIDENCE BREAKDOWN",
            "> [!NOTE]",
            "> All confidence scores distinguish raw model probabilities from calibrated likelihoods and graph scores. "
            "> Probabilities are never presented as proofs of factual authenticity.",
            "",
            "| Transition | Type | Raw Probability | Calibrated Probability | Graph Score | Optimization Decision |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        for t in provenance.transitions:
            t_type = "GAP_TRANSITION" if t.is_gap_transition else "DIRECT_SUCCESSOR"
            dec_str = "SELECTED" if t.optimization_decision == 1 else "REJECTED"
            lines.append(
                f"| `{t.source_fragment_id}` -> `{t.target_fragment_id}` | `{t_type}` | "
                f"`{t.raw_probability:.4f}` | `{t.calibrated_probability:.4f}` | "
                f"`{t.graph_edge_score:.4f}` | `{dec_str}` |"
            )

        if uncertainty:
            lines.extend([
                "",
                "### Multidimensional Uncertainty Signals",
                f"- **Predictive Entropy:** `{uncertainty.predictive_entropy:.4f}` (bits)",
                f"- **Model Disagreement (Variance):** `{uncertainty.ensemble_variance:.4f}`",
                f"- **Alternative Path Margin:** `{uncertainty.alternative_margin:.4f}`",
                f"- **Graph Topological Ambiguity:** `{uncertainty.graph_ambiguity:.4f}`",
                f"- **Missing Gap Ratio:** `{uncertainty.gap_penalty_ratio:.2f}`",
                f"- **Composite Uncertainty Score:** **`{uncertainty.composite_uncertainty_score:.4f}`** (Scale: 0.0=Certain to 1.0=Uncertain)",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## 4. FORMAT STRUCTURAL & PARSER VALIDATION AUDIT",
            f"- **Format Validator Executed:** `{val.get('format_name', 'UNKNOWN')}`",
            f"- **Highest Verified Tier:** `{val_level}`",
            f"- **Structural Syntax Passed (L2):** `{val.get('is_structurally_valid', False)}`",
            f"- **Decompression / Renderable (L3):** `{val.get('is_fully_renderable', False)}`",
            f"- **Reference Ground Truth Match (L5):** `{val.get('has_ground_truth_match', False)}`",
            "",
            "### Validator Disclosures and Warnings:",
        ])

        errs = val.get("errors", [])
        warns = val.get("warnings", [])

        if not errs and not warns:
            lines.append("- *No structural errors or warnings reported by independent parser.*")
        else:
            for e in errs:
                lines.append(f"- **ERROR:** {e}")
            for w in warns:
                lines.append(f"- **WARNING:** {w}")

        meta = val.get("extracted_metadata", {})
        if meta:
            lines.extend([
                "",
                "### Parsed Technical Metadata:",
                "```json",
                json.dumps(meta, indent=2),
                "```",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## 5. UNRESOLVED GAPS & MISSING EVIDENCE DISCLOSURE",
        ])

        if not provenance.missing_gaps:
            lines.append("- *Zero unresolved gaps detected. File is contiguous or completely bridged.*")
        else:
            lines.append(f"- **Detected Unresolved Missing Gaps:** {len(provenance.missing_gaps)}")
            for g in provenance.missing_gaps:
                lines.append(f"  - Position {g.get('position')}: Placeholder `{g.get('gap_marker')}`")

        if rejected_hypotheses:
            lines.extend([
                "",
                "---",
                "",
                "## 6. REJECTED COMPETING HYPOTHESES",
                "| Hypothesis ID | Fragment Sequence | Score | Rejection Reason |",
                "| :--- | :--- | :---: | :--- |",
            ])
            for hyp in rejected_hypotheses:
                seq_str = " -> ".join(hyp.get("sequence", [])[:4]) + ("..." if len(hyp.get("sequence", [])) > 4 else "")
                lines.append(
                    f"| `{hyp.get('id', 'hyp_alt')}` | `{seq_str}` | "
                    f"`{hyp.get('score', 0.0):.4f}` | {hyp.get('reason', 'Lower evidentiary support')} |"
                )

        lines.extend([
            "",
            "---",
            "",
            "## 7. CRYPTOGRAPHIC PROVENANCE & AUDIT REPRODUCIBILITY",
            f"- **Provenance Record Hash:** `{provenance.provenance_hash}`",
            "- **Reproducibility Statement:** *This reconstruction can be bit-for-bit reproduced by executing the ForenSight optimization pipeline against the specified input fragment hashes under software version " + provenance.software_version + ".*",
            "",
            "> [!IMPORTANT]",
            "> **MANDATORY SCIENTIFIC & FORENSIC DISCLAIMER:**",
            "> Machine-learning predictions and probability estimates generated by ForenSight reflect statistical likelihoods "
            "> derived from trained models and observable evidence. They do **NOT** constitute definitive proof of factual "
            "> authenticity, file origin, or authorship. All reconstructed digital evidence must be independently corroborated "
            "> by qualified digital forensics examiners prior to presentation in legal proceedings.",
        ])

        return "\n".join(lines)
