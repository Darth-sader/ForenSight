"""Missing Fragment and Gap Penalty Modeling (v2.0.0-phase5).

Provides formal mathematical modeling for non-consecutive fragment continuity:
1. Distinguishes direct transitions (x_ij = 1) from gap transitions (g_ij = 1).
2. Computes calibrated gap penalty gamma_gap(i, j) based on:
   - Base gap occurrence cost gamma_0
   - Physical offset span (when observable)
   - Information entropy discontinuity |H_i - H_j|
   - Format structural boundary mismatch
3. Assigns objective weights: w_gap(i, j) = w_next(i, j) - gamma_gap(i, j)
4. Reconstructs missing data as placeholder blocks (zero-byte or marked fills)
   preserving sequence alignment without corrupting adjacent byte offsets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from phase1_leakage_free.fragment_schema_v2 import ObservableFragment


@dataclass
class GapEdgeInfo:
    """Attributes of a candidate missing-fragment gap edge."""
    source_idx: int
    target_idx: int
    base_weight: float                  # e.g., logit(P_next) or logit(P_same)
    gap_penalty: float                  # Computed penalty gamma_gap >= 0
    net_weight: float                   # base_weight - gap_penalty
    estimated_missing_blocks: int = 1   # Estimated missing fragments between i and j
    estimated_missing_bytes: int = 512  # Estimated missing byte count


class MissingFragmentModel:
    """Computes missing fragment gap penalties and generates gap candidate edges."""

    def __init__(
        self,
        base_gap_penalty: float = 2.5,        # Base logit penalty for postulating a gap
        entropy_weight: float = 0.5,          # Penalty weight for entropy divergence
        offset_distance_weight: float = 0.05, # Penalty per missing 4KB block span
        default_block_size: int = 4096,       # Standard forensic cluster size
        max_missing_blocks: int = 32,         # Maximum plausible missing span
        min_same_file_prob: float = 0.35,     # Minimum P_same required to allow a gap
    ):
        self.base_gap_penalty = base_gap_penalty
        self.entropy_weight = entropy_weight
        self.offset_distance_weight = offset_distance_weight
        self.default_block_size = default_block_size
        self.max_missing_blocks = max_missing_blocks
        self.min_same_file_prob = min_same_file_prob

    def compute_fragment_entropy(self, data: bytes) -> float:
        """Shannon entropy in bits per byte [0.0, 8.0]."""
        if not data:
            return 0.0
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probs = counts[counts > 0] / len(data)
        return float(-np.sum(probs * np.log2(probs)))

    def compute_gap_penalty(
        self,
        frag_i: ObservableFragment,
        frag_j: ObservableFragment,
        p_same: float,
        p_next: float,
    ) -> Tuple[float, int, int]:
        """Compute gamma_gap(i, j), estimated missing block count, and missing bytes.

        Mathematical Formulation:
            gamma_gap(i, j) = gamma_0
                            + w_ent * |H(i) - H(j)|
                            + w_dist * min(k_est, k_max)
                            + max(0.0, logit(tau_min) - logit(P_same))
        """
        # 1. Base cost
        penalty = self.base_gap_penalty

        # 2. Information entropy divergence
        ent_i = self.compute_fragment_entropy(frag_i.fragment_bytes)
        ent_j = self.compute_fragment_entropy(frag_j.fragment_bytes)
        delta_ent = abs(ent_i - ent_j)
        penalty += self.entropy_weight * delta_ent

        # 3. Physical distance span (if observed offsets are nonzero and increasing)
        est_blocks = 1
        est_bytes = self.default_block_size
        if frag_j.observed_offset > frag_i.observed_offset + frag_i.fragment_length:
            delta_bytes = frag_j.observed_offset - (frag_i.observed_offset + frag_i.fragment_length)
            est_blocks = max(1, min(self.max_missing_blocks, int(math.ceil(delta_bytes / self.default_block_size))))
            est_bytes = delta_bytes
            penalty += self.offset_distance_weight * est_blocks

        # 4. Same-file confidence barrier
        if p_same < self.min_same_file_prob:
            # Harsh penalty if same-file probability is below threshold
            penalty += 10.0 * (self.min_same_file_prob - p_same)

        return float(penalty), est_blocks, est_bytes

    def evaluate_gap_candidate(
        self,
        u: int,
        v: int,
        frag_u: ObservableFragment,
        frag_v: ObservableFragment,
        p_same: float,
        p_next: float,
        logit_same_weight: float,
    ) -> Optional[GapEdgeInfo]:
        """Evaluate whether a gap edge (u, v) is plausible and calculate net weight.
        
        Gap edge weight is derived from same-file affinity penalized by gap cost:
            w_gap(u, v) = logit(P_same(u, v)) - gamma_gap(u, v)
        """
        # A gap should only be considered if fragments have reasonable same-file affinity
        if p_same < self.min_same_file_prob:
            return None

        gamma, est_blocks, est_bytes = self.compute_gap_penalty(frag_u, frag_v, p_same, p_next)

        # Net weight: same-file affinity minus gap penalty
        net_w = logit_same_weight - gamma

        return GapEdgeInfo(
            source_idx=u,
            target_idx=v,
            base_weight=logit_same_weight,
            gap_penalty=gamma,
            net_weight=net_w,
            estimated_missing_blocks=est_blocks,
            estimated_missing_bytes=est_bytes,
        )

    @staticmethod
    def create_gap_placeholder(num_bytes: int = 4096) -> bytes:
        """Generate null-byte placeholder representing missing intermediate data."""
        return b"\x00" * max(1, num_bytes)
