"""Forensic constraints (doc §22): deterministic validation applied AFTER ML
inference.  "ML proposes. Constraints validate."  The ML model is never
allowed to blindly declare forensic truth."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from features import FeatureEngine
from features.structural_features import FORMATS, OFFICE_TYPES
from graph.graph_inference import CandidatePath


class ForensicConstraints:
    def __init__(self, engine: FeatureEngine, fragment_size: int = 512,
                 file_size_stats: Optional[Dict[str, Dict[str, float]]] = None,
                 image_size_blocks: int = 0):
        self.engine = engine
        self.fragment_size = fragment_size
        self.size_stats = file_size_stats or {}
        self.image_size_blocks = image_size_blocks

    # ------------------------------------------------------------------
    def check(self, path: CandidatePath, claimed_type: Optional[str],
              used_elsewhere: Optional[set] = None) -> Tuple[int, List[str], Dict[str, object]]:
        """Return (violations, issues, forensic_evidence) for a candidate path."""
        issues: List[str] = []
        violations = 0
        nodes = path.nodes
        first, last = nodes[0], nodes[-1]
        s_first = self.engine.struct_dict(first)
        s_last = self.engine.struct_dict(last)

        evidence: Dict[str, object] = {"claimed_type": claimed_type or "UNKNOWN"}

        # --- header / footer rules (only when a type is claimed)
        header_ok = footer_ok = None
        if claimed_type and claimed_type in FORMATS:
            header_ok = s_first.get(f"header_{claimed_type}", 0.0) >= 0.5
            if not header_ok:
                violations += 1
                issues.append(f"Missing valid {claimed_type} header evidence on first fragment")
            footer_marker = "footer_" + claimed_type
            footer_ok = s_last.get(footer_marker, 0.0) >= 0.5
            if not footer_ok:
                issues.append("No valid footer on final fragment (partial file possible)")
            # a footer inside the middle of the sequence means the file ended early
            for n in nodes[1:-1]:
                if self.engine.struct_dict(n).get(footer_marker, 0.0) >= 1.0:
                    violations += 1
                    issues.append("Footer marker found mid-sequence (file ended early)")
                    break
            # internal marker evidence somewhere in the sequence
            internal_hits = sum(
                1.0 for n in nodes
                if any(self.engine.struct_dict(n).get(f"internal_{t}", 0.0) > 0
                       for t in self._sibling_types(claimed_type)))
            if internal_hits == 0 and len(nodes) > 2:
                issues.append("No internal structural markers for claimed format")
        evidence["header_valid"] = bool(header_ok) if header_ok is not None else None
        evidence["footer_valid"] = bool(footer_ok) if footer_ok is not None else None

        # --- no impossible fragment reuse
        if len(set(nodes)) != len(nodes):
            violations += 2
            issues.append("Duplicate fragment reuse within candidate")
        reuse = 0
        if used_elsewhere:
            reuse = len(set(nodes) & used_elsewhere)
            if reuse:
                issues.append(f"Competing fragment assignment ({reuse} fragments shared)")
        evidence["no_impossible_reuse"] = reuse == 0

        # --- fragment order / physical plausibility
        offsets = [float(self.engine.metas_[n].get("physical_offset") or 0) for n in nodes]
        if any(offsets[i] > offsets[i + 1] for i in range(len(offsets) - 1)):
            violations += 1
            issues.append("Fragment order violates physical write order")
        max_jump = self.fragment_size * max(64, self.image_size_blocks // 2)
        if any(offsets[i + 1] - offsets[i] > max_jump for i in range(len(offsets) - 1)):
            issues.append("Implausibly large physical gap between fragments")

        # --- filesystem constraints
        unalloc = sum(1 for n in nodes
                      if str(self.engine.metas_[n].get("allocation_status")) == "unallocated")
        if unalloc:
            issues.append(f"{unalloc} fragments from unallocated/slack space")
        evidence["filesystem_consistent"] = unalloc == 0

        # --- format-specific consistency across members
        consist = [float(self.engine.struct_dict(n).get("format_consistency", 0.5)) for n in nodes]
        evidence["structural_consistency"] = float(np.mean(consist))
        conflicts = sum(1 for n in nodes if self.engine.struct_dict(n).get("multi_format_conflict", 0) > 0)
        if conflicts:
            issues.append(f"{conflicts} fragments with multi-format conflicts")

        # --- expected file-size range
        est_size = sum(int(self.engine.metas_[n].get("fragment_size") or 0) for n in nodes)
        evidence["estimated_size"] = est_size
        if claimed_type in self.size_stats:
            med = self.size_stats[claimed_type]["median"]
            if est_size < 0.25 * med or est_size > 4.0 * med:
                issues.append(f"Estimated size {est_size} outside typical {claimed_type} range")
        evidence["size_in_typical_range"] = True if claimed_type not in self.size_stats else bool(
            0.25 * self.size_stats[claimed_type]["median"] <= est_size <= 4.0 * self.size_stats[claimed_type]["median"])
        return violations, issues, evidence

    @staticmethod
    def _sibling_types(t: str) -> List[str]:
        if t in OFFICE_TYPES:
            return ["ZIP", "DOCX", "XLSX"]
        return [t]
