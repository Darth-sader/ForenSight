"""Multi-Evidence Forensic Candidate Generator (v2.0.0-phase4).

Generates directed candidate successor transitions (u -> v) using:
1. Deep embedding / single-feature nearest neighbors (cosine similarity).
2. Byte-level statistical similarity (histogram L1, intersection).
3. Boundary byte compatibility (continuity from end of u to start of v).
4. Format compatibility and structural marker constraints.
5. Observable physical disk context (when legitimately available).

Strictly isolates candidate generation from ground-truth logical positions.
Evaluates graph topology, density, isolated nodes, and Recall@K.
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

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from features.structural_features import detect_format_markers
from phase1_leakage_free.fragment_schema_v2 import ObservableFragment


@dataclass
class CandidateGenerationMetrics:
    """Metrics assessing candidate generator quality and topological efficiency."""
    k: int
    total_fragments: int
    total_edges: int
    graph_density: float
    avg_in_degree: float
    avg_out_degree: float
    isolated_nodes: int
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    generation_runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "total_fragments": self.total_fragments,
            "total_edges": self.total_edges,
            "graph_density": round(self.graph_density, 6),
            "avg_in_degree": round(self.avg_in_degree, 4),
            "avg_out_degree": round(self.avg_out_degree, 4),
            "isolated_nodes": self.isolated_nodes,
            "recall_at_k": {f"recall@{k}": round(v, 4) for k, v in self.recall_at_k.items()},
            "precision_at_k": {f"precision@{k}": round(v, 4) for k, v in self.precision_at_k.items()},
            "runtime_seconds": round(self.generation_runtime_seconds, 4),
        }


class CandidateGeneratorV2:
    """Uncertainty-aware, multi-evidence candidate generator."""

    def __init__(
        self,
        use_embeddings: bool = True,
        use_boundary: bool = True,
        use_physical_context: bool = False,
        seed: int = 42,
    ):
        self.use_embeddings = use_embeddings
        self.use_boundary = use_boundary
        self.use_physical_context = use_physical_context
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _compute_boundary_score(self, b_u: bytes, b_v: bytes) -> float:
        """Calculate byte-continuity smoothness between end of u and start of v."""
        if not self.use_boundary or len(b_u) < 8 or len(b_v) < 8:
            return 0.5
        tail = np.frombuffer(b_u[-8:], dtype=np.uint8)
        head = np.frombuffer(b_v[:8], dtype=np.uint8)
        # Low gradient difference indicates continuity
        diff = np.mean(np.abs(np.diff(np.concatenate([tail[-4:], head[:4]]))))
        return float(1.0 / (1.0 + diff / 32.0))

    def _check_format_compatibility(
        self,
        struct_u: Dict[str, float],
        struct_v: Dict[str, float],
    ) -> float:
        """Check format compatibility between adjacent candidate fragments."""
        # A footer fragment in u should not generally have an internal body successor
        is_footer_u = struct_u.get("is_footer", 0.0) > 0.5
        is_header_v = struct_v.get("is_header", 0.0) > 0.5

        if is_footer_u and not is_header_v:
            # Transition out of a footer into a non-header is unlikely
            return 0.1
        if is_header_v:
            # Header inside stream is unlikely unless multi-part archive
            return 0.2
        return 1.0

    def generate_candidate_graph(
        self,
        fragments: List[ObservableFragment],
        feature_matrix: Optional[np.ndarray] = None,  # (N, D) single fragment features
        embeddings: Optional[np.ndarray] = None,      # (N, E) deep metric embeddings
        fixed_k: int = 8,
        true_successors: Optional[Dict[str, str]] = None,  # Optional GT for Recall@K evaluation
    ) -> Tuple[nx.DiGraph, np.ndarray, CandidateGenerationMetrics]:
        """Generate directed candidate transition graph for a pool of observable fragments.

        Args:
            fragments: List of ObservableFragment objects.
            feature_matrix: Precomputed single-fragment features (histograms, entropy, etc.).
            embeddings: Deep metric learning embeddings.
            fixed_k: Target number of candidate successors per fragment.
            true_successors: Ground-truth successor mapping (u_id -> true_v_id) for metrics ONLY.

        Returns:
            (candidate_digraph, candidate_pair_indices, metrics)
        """
        t0 = time.time()
        n = len(fragments)
        frag_ids = [f.fragment_id for f in fragments]
        frag_map = {f.fragment_id: i for i, f in enumerate(fragments)}

        if n <= 1:
            g = nx.DiGraph()
            for f in fragments:
                g.add_node(f.fragment_id)
            metrics = CandidateGenerationMetrics(
                k=fixed_k, total_fragments=n, total_edges=0, graph_density=0.0,
                avg_in_degree=0.0, avg_out_degree=0.0, isolated_nodes=n,
                recall_at_k={k: 0.0 for k in (1, 3, 5, fixed_k)},
                precision_at_k={k: 0.0 for k in (1, 3, 5, fixed_k)},
                generation_runtime_seconds=time.time() - t0,
            )
            return g, np.zeros((0, 2), dtype=np.int64), metrics

        # 1. Similarity representations
        if self.use_embeddings and embeddings is not None and len(embeddings) == n:
            rep = embeddings
        elif feature_matrix is not None and len(feature_matrix) == n:
            rep = feature_matrix
        else:
            # Compute byte histogram representations on the fly
            hists = np.array([
                np.bincount(np.frombuffer(f.fragment_bytes, dtype=np.uint8), minlength=256)
                for f in fragments
            ], dtype=np.float32)
            rep = hists

        # Normalize representations for cosine similarity
        norms = np.linalg.norm(rep, axis=1, keepdims=True) + 1e-8
        rep_norm = rep / norms
        sim_matrix = rep_norm @ rep_norm.T
        np.fill_diagonal(sim_matrix, -1e6)

        # 2. Extract structural markers for format compatibility
        struct_markers = [
            detect_format_markers(np.frombuffer(f.fragment_bytes, dtype=np.uint8))
            for f in fragments
        ]

        candidate_pairs_set: Set[Tuple[int, int]] = set()
        k_eval_list = [1, 3, 5, fixed_k]
        recall_hits = {k: 0 for k in k_eval_list}
        precision_hits = {k: 0 for k in k_eval_list}
        total_gt_pairs = 0

        # Build candidate edges per anchor fragment
        for i in range(n):
            u_frag = fragments[i]
            u_id = u_frag.fragment_id
            u_bytes = u_frag.fragment_bytes
            u_struct = struct_markers[i]

            # Similarity scores
            sim_scores = sim_matrix[i].copy()

            # Rank top candidate indices
            top_indices = np.argsort(-sim_scores)[:min(fixed_k * 3, n - 1)]

            scored_candidates: List[Tuple[int, float]] = []
            for j in top_indices:
                if i == j:
                    continue
                v_frag = fragments[j]
                v_bytes = v_frag.fragment_bytes
                v_struct = struct_markers[j]

                # Evidence fusion
                emb_sim = float(sim_scores[j])
                b_score = self._compute_boundary_score(u_bytes, v_bytes)
                fmt_compat = self._check_format_compatibility(u_struct, v_struct)

                phys_score = 0.5
                if self.use_physical_context:
                    # Legitimate observable LBA distance in disk image
                    lba_diff = abs(u_frag.observed_offset - v_frag.observed_offset)
                    phys_score = float(1.0 / (1.0 + lba_diff / 4096.0))

                # Composite heuristic candidate score
                cand_score = (0.50 * emb_sim) + (0.30 * b_score) + (0.10 * fmt_compat) + (0.10 * phys_score)
                scored_candidates.append((j, cand_score))

            # Sort candidate successors by composite score descending
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            chosen = [c[0] for c in scored_candidates[:fixed_k]]

            for j in chosen:
                candidate_pairs_set.add((i, j))

            # Ground-truth evaluation (if provided)
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

        # 3. Construct NetworkX Directed Graph
        g = nx.DiGraph()
        for f in fragments:
            g.add_node(
                f.fragment_id,
                observed_offset=f.observed_offset,
                length=f.fragment_length,
                allocation=f.allocation_status,
            )

        for src, dst in candidate_pairs:
            u_id = frag_ids[src]
            v_id = frag_ids[dst]
            sim_val = float(sim_matrix[src, dst])
            b_val = self._compute_boundary_score(fragments[src].fragment_bytes, fragments[dst].fragment_bytes)
            g.add_edge(
                u_id,
                v_id,
                embedding_sim=sim_val,
                boundary_compat=b_val,
                source=u_id,
                target=v_id,
            )

        elapsed = time.time() - t0

        # Topological diagnostics
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

        metrics = CandidateGenerationMetrics(
            k=fixed_k,
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

        return g, candidate_pairs, metrics
