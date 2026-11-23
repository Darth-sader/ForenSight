"""Format-Aware Constraints and Forensic Boundary Extractor (v2.0.0-phase5).

Translates file format structural evidence into hard MILP/CP-SAT constraints
and soft objective boundary bonuses:
1. Header Hard Constraint: In-degree = 0 (no predecessor permitted for file headers).
2. Footer Hard Constraint: Out-degree = 0 (no successor permitted for file footers).
3. Incompatible Format Exclusion: x_ij = 0, g_ij = 0 when fragments exhibit contradictory formats.
4. Seed/Tail Soft Bonuses: rho_head * s_i and rho_tail * t_i to reward valid start and end points.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from features.structural_features import detect_format_markers
from phase1_leakage_free.fragment_schema_v2 import ObservableFragment


@dataclass
class FormatEvidence:
    """Format classification and marker detections for a single fragment."""
    fragment_idx: int
    detected_format: Optional[str]
    is_header: bool
    is_footer: bool
    header_confidence: float
    footer_confidence: float
    internal_marker_count: int


class FormatConstraintExtractor:
    """Extracts forensic format boundaries, incompatibilities, and objective bonuses."""

    def __init__(
        self,
        header_threshold: float = 0.8,
        footer_threshold: float = 0.8,
        head_bonus: float = 2.0,
        tail_bonus: float = 2.0,
    ):
        self.header_threshold = header_threshold
        self.footer_threshold = footer_threshold
        self.head_bonus = head_bonus
        self.tail_bonus = tail_bonus

    def analyze_fragment(self, idx: int, frag: ObservableFragment) -> FormatEvidence:
        """Analyze a fragment's bytes for structural markers."""
        raw_bytes = frag.fragment_bytes
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        feats = detect_format_markers(arr)

        detected_fmt = None
        max_h_score = 0.0
        for fmt in ["JPEG", "PNG", "PDF", "ZIP", "DOCX", "XLSX", "MP3", "MP4"]:
            h_score = feats.get(f"header_{fmt}", 0.0)
            if h_score > max_h_score:
                max_h_score = h_score
                detected_fmt = fmt

        max_f_score = 0.0
        for fmt in ["JPEG", "PNG", "PDF", "ZIP", "DOCX", "XLSX", "MP3", "MP4"]:
            f_score = feats.get(f"footer_{fmt}", 0.0)
            if f_score > max_f_score:
                max_f_score = f_score
                if detected_fmt is None:
                    detected_fmt = fmt

        is_h = max_h_score >= self.header_threshold
        is_f = max_f_score >= self.footer_threshold

        return FormatEvidence(
            fragment_idx=idx,
            detected_format=detected_fmt if (is_h or is_f or feats.get("marker_density", 0.0) > 0.3) else None,
            is_header=is_h,
            is_footer=is_f,
            header_confidence=max_h_score,
            footer_confidence=max_f_score,
            internal_marker_count=int(feats.get("marker_density", 0.0) * 6),
        )

    def extract_boundaries(
        self,
        fragments: Sequence[ObservableFragment],
    ) -> Tuple[Set[int], Set[int], Dict[int, FormatEvidence]]:
        """Identify all confirmed header fragments, footer fragments, and format evidence."""
        header_nodes: Set[int] = set()
        footer_nodes: Set[int] = set()
        evidence_map: Dict[int, FormatEvidence] = {}

        for i, frag in enumerate(fragments):
            ev = self.analyze_fragment(i, frag)
            evidence_map[i] = ev
            if ev.is_header:
                header_nodes.add(i)
            if ev.is_footer:
                footer_nodes.add(i)

        return header_nodes, footer_nodes, evidence_map

    def find_incompatible_pairs(
        self,
        fragments: Sequence[ObservableFragment],
        evidence_map: Dict[int, FormatEvidence],
    ) -> Set[Tuple[int, int]]:
        """Identify pairs of fragments with mutually incompatible format markers."""
        incompatible: Set[Tuple[int, int]] = set()
        n = len(fragments)

        # Incompatible families: e.g. JPEG vs PNG vs PDF vs ZIP/Office
        office_group = {"ZIP", "DOCX", "XLSX"}

        for i in range(n):
            ev_i = evidence_map.get(i)
            if ev_i is None or ev_i.detected_format is None:
                continue

            for j in range(n):
                if i == j:
                    continue
                ev_j = evidence_map.get(j)
                if ev_j is None or ev_j.detected_format is None:
                    continue

                fmt_i = ev_i.detected_format
                fmt_j = ev_j.detected_format

                # Check if formats are contradictory
                # If one is JPEG and other is PNG/PDF/Office
                is_office_i = fmt_i in office_group
                is_office_j = fmt_j in office_group

                if is_office_i and is_office_j:
                    # DOCX / XLSX / ZIP share container markers, do not mark incompatible
                    continue
                elif fmt_i != fmt_j:
                    # Require strong evidence before hard-excluding
                    if (ev_i.is_header or ev_i.is_footer or ev_i.internal_marker_count >= 2) and \
                       (ev_j.is_header or ev_j.is_footer or ev_j.internal_marker_count >= 2):
                        incompatible.add((i, j))

        return incompatible

    def compute_seed_tail_bonuses(
        self,
        num_nodes: int,
        evidence_map: Dict[int, FormatEvidence],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute linear objective coefficients for s_i (seeds) and t_i (tails)."""
        s_bonus = np.zeros(num_nodes, dtype=float)
        t_bonus = np.zeros(num_nodes, dtype=float)

        for i in range(num_nodes):
            ev = evidence_map.get(i)
            if ev is not None:
                if ev.is_header:
                    s_bonus[i] = self.head_bonus * ev.header_confidence
                if ev.is_footer:
                    t_bonus[i] = self.tail_bonus * ev.footer_confidence

        return s_bonus, t_bonus
