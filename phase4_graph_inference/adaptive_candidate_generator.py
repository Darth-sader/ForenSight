"""Uncertainty-Aware Adaptive Candidate Generator (v2.0.0-phase4).

Dynamically allocates candidate successor degree K(F_i) in [K_min, K_max] based on:
1. Top-2 Candidate Score Margin: Delta_12 = s_1 - s_2 (tie-breaking ambiguity).
2. Predictive Shannon Entropy: H(F_i) = -sum p_j log2(p_j) (probabilistic spread).
3. Embedding Neighborhood Density: Local clustering tightness.
4. Format / Structural Boundary Ambiguity.

Enables selective candidate expansion under high uncertainty and candidate pruning under high confidence.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from features.structural_features import detect_format_markers
from phase1_leakage_free.fragment_schema_v2 import ObservableFragment
from phase4_graph_inference.candidate_generator_v2 import CandidateGenerationMetrics


@dataclass
class AdaptiveUncertaintyProfile:
    """Uncertainty indicators computed for an anchor fragment."""
    fragment_id: str
    top2_margin: float
    predictive_entropy: float
    local_density: float
    allocated_k: int


class AdaptiveCandidateGenerator:
    """Configurable adaptive candidate generation scaling K with uncertainty."""

    def __init__(
        self,
        k_min: int = 4,
        k_base: int = 8,
        k_max: int = 16,
        alpha_entropy: float = 1.0,
        alpha_margin: float = 1.0,
        temperature: float = 0.25,
        use_boundary: bool = True,
        use_physical_context: bool = False,
        seed: int = 42,
    ):
        self.k_min = k_min
        self.k_base = k_base
        self.k_max = k_max
        self.alpha_entropy = alpha_entropy
        self.alpha_margin = alpha_margin
        self.temperature = temperature
        self.use_boundary = use_boundary
        self.use_physical_context = use_physical_context
        self.seed = seed

    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute normalized Shannon entropy in [0, 1]."""
        p = probs[probs > 1e-9]
        if len(p) <= 1:
            return 0.0
        h = -np.sum(p * np.log2(p))
        max_h = np.log2(len(probs))
        return float(h / max_h) if max_h > 0 else 0.0

    def compute_fragment_uncertainty(
        self,
        candidate_scores: np.ndarray,
    ) -> Tuple[float, float, int]:
        """Compute top-2 margin, normalized entropy, and dynamic K allocation.

        Returns:
            (top2_margin, normalized_entropy, allocated_k)
        """
        if len(candidate_scores) < 2:
            return 1.0, 0.0, self.k_min

        sorted_scores = np.sort(candidate_scores)[::-1]
        top1 = sorted_scores[0]
        top2 = sorted_scores[1]
        margin = float(max(0.0, top1 - top2))

        # Softmax distribution over top candidate pool
        top_pool = sorted_scores[:min(len(sorted_scores), self.k_max * 2)]
        exp_s = np.exp((top_pool - np.max(top_pool)) / self.temperature)
        probs = exp_s / (np.sum(exp_s) + 1e-12)
        entropy = self._compute_entropy(probs)

        # Dynamic K allocation formula:
        # Expands toward k_max when entropy is high; contracts toward k_min when margin is high
        delta_k = (self.alpha_entropy * entropy * (self.k_max - self.k_base)) - (
            self.alpha_margin * margin * (self.k_base - self.k_min)
        )
        allocated = int(np.clip(round(self.k_base + delta_k), self.k_min, self.k_max))

        return margin, entropy, allocated

    def generate_adaptive_candidate_graph(
        self,
        fragments: List[ObservableFragment],
        feature_matrix: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        true_successors: Optional[Dict[str, str]] = None,
    ) -> Tuple[nx.DiGraph, np.ndarray, CandidateGenerationMetrics, List[AdaptiveUncertaintyProfile]]:
        """Generate directed candidate graph using adaptive uncertainty-budgeted K."""
        t0 = time.time()
        n = len(fragments)
        frag_ids = [f.fragment_id for f in fragments]
        frag_map = {f.fragment_id: i for i, f in enumerate(fragments)}

        if n <= 1:
            g = nx.DiGraph()
            for f in fragments:
                g.add_node(f.fragment_id)
            metrics = CandidateGenerationMetrics(
                k=self.k_base, total_fragments=n, total_edges=0, graph_density=0.0,
                avg_in_degree=0.0, avg_out_degree=0.0, isolated_nodes=n,
                recall_at_k={k: 0.0 for k in (1, 3, 5, self.k_base, self.k_max)},
                precision_at_k={k: 0.0 for k in (1, 3, 5, self.k_base, self.k_max)},
                generation_runtime_seconds=time.time() - t0,
            )
            return g, np.zeros((0, 2), dtype=np.int64), metrics, []

        # Representations
        if embeddings is not None and len(embeddings) == n:
            rep = embeddings
        elif feature_matrix is not None and len(feature_matrix) == n:
            rep = feature_matrix
        else:
            hists = np.array([
                np.bincount(np.frombuffer(f.fragment_bytes, dtype=np.uint8), minlength=256)
                for f in fragments
            ], dtype=np.float32)
            rep = hists

        norms = np.linalg.norm(rep, axis=1, keepdims=True) + 1e-8
        rep_norm = rep / norms
        sim_matrix = rep_norm @ rep_norm.T
        np.fill_diagonal(sim_matrix, -1e6)

        struct_markers = [detect_format_markers(np.frombuffer(f.fragment_bytes, dtype=np.uint8)) for f in fragments]

        candidate_pairs_set: Set[Tuple[int, int]] = set()
        profiles: List[AdaptiveUncertaintyProfile] = []

        k_eval_list = [1, 3, 5, self.k_base, self.k_max]
        recall_hits = {k: 0 for k in k_eval_list}
        precision_hits = {k: 0 for k in k_eval_list}
        total_gt_pairs = 0

        for i in range(n):
            u_frag = fragments[i]
            u_id = u_frag.fragment_id
            u_bytes = u_frag.fragment_bytes
            u_struct = struct_markers[i]

            sim_scores = sim_matrix[i]
            top_pool_indices = np.argsort(-sim_scores)[:min(self.k_max * 3, n - 1)]

            scored_candidates: List[Tuple[int, float]] = []
            for j in top_pool_indices:
                if i == j:
                    continue
                v_frag = fragments[j]
                v_bytes = v_frag.fragment_bytes
                v_struct = struct_markers[j]

                emb_sim = float(sim_scores[j])
                b_diff = np.mean(np.abs(np.diff(np.concatenate([
                    np.frombuffer(u_bytes[-8:], dtype=np.uint8)[-4:],
                    np.frombuffer(v_bytes[:8], dtype=np.uint8)[:4],
                ])))) if self.use_boundary else 32.0
                b_score = float(1.0 / (1.0 + b_diff / 32.0))

                fmt_compat = 1.0
                if u_struct.get("is_footer", 0.0) > 0.5 and v_struct.get("is_header", 0.0) < 0.5:
                    fmt_compat = 0.1
                elif v_struct.get("is_header", 0.0) > 0.5:
                    fmt_compat = 0.2

                cand_score = (0.50 * emb_sim) + (0.35 * b_score) + (0.15 * fmt_compat)
                scored_candidates.append((j, cand_score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            cand_scores_arr = np.array([c[1] for c in scored_candidates])

            # Compute uncertainty profile and dynamic K allocation
            margin, entropy, alloc_k = self.compute_fragment_uncertainty(cand_scores_arr)
            density_count = float(np.sum(sim_scores > 0.8))

            profile = AdaptiveUncertaintyProfile(
                fragment_id=u_id,
                top2_margin=margin,
                predictive_entropy=entropy,
                local_density=density_count,
                allocated_k=alloc_k,
            )
            profiles.append(profile)

            # Keep top alloc_k candidates for this anchor fragment
            chosen = [c[0] for c in scored_candidates[:alloc_k]]
            for j in chosen:
                candidate_pairs_set.add((i, j))

            # GT evaluation
            if true_successors and u_id in true_successors:
                target_v_id = true_successors[u_id]
                if target_v_id in frag_map:
                    target_j = frag_map[target_v_id]
                    total_gt_pairs += 1
                    ranked_j = [c[0] for c in scored_candidates]
                    for k in k_eval_list:
                        if target_j in ranked_j[:k]:
                            recall_hits[k] += 1
                            precision_hits[k] += 1 / k

        candidate_pairs = np.array(list(candidate_pairs_set), dtype=np.int64)

        g = nx.DiGraph()
        for f in fragments:
            g.add_node(f.fragment_id, observed_offset=f.observed_offset, length=f.fragment_length)

        for src, dst in candidate_pairs:
            u_id = frag_ids[src]
            v_id = frag_ids[dst]
            g.add_edge(u_id, v_id, embedding_sim=float(sim_matrix[src, dst]), source=u_id, target=v_id)

        elapsed = time.time() - t0
        total_edges = g.number_of_edges()
        density = nx.density(g) if n > 1 else 0.0
        in_degs = [d for _, d in g.in_degree()]
        out_degs = [d for _, d in g.out_degree()]
        avg_in = float(np.mean(in_degs)) if in_degs else 0.0
        avg_out = float(np.mean(out_degs)) if out_degs else 0.0
        isolated = sum(1 for _, d in g.degree() if d == 0)

        recall_at_k = {
            k: (recall_hits[k] / total_gt_pairs if total_gt_pairs > 0 else 0.0)
            for k in k_eval_list
        }
        precision_at_k = {
            k: (precision_hits[k] / total_gt_pairs if total_gt_pairs > 0 else 0.0)
            for k in k_eval_list
        }

        avg_allocated_k = int(round(np.mean([p.allocated_k for p in profiles]))) if profiles else self.k_base

        metrics = CandidateGenerationMetrics(
            k=avg_allocated_k,
            total_fragments=n,
            total_edges=total_edges,
            graph_density=density,
            avg_in_degree=avg_in,
            avg_out_degree=avg_out,
            isolated_nodes=isolated,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            generation_runtime_seconds=elapsed,
        )

        return g, candidate_pairs, metrics, profiles
