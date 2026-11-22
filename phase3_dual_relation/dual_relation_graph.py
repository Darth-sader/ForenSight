"""Dual-Relation Graph Construction and Multi-File Reconstruction Engine (v2.0.0-phase3).

Integrates symmetric file membership P_same and directed transition P_next into
a structured candidate graph:
1. P_same governs global file membership and partitioning (clustering).
2. P_next governs local directed logical sequencing (path traversal).
3. P_prev ensures bidirectional consensus and detects dead-end transitions.
4. Every directed edge retains all 9 mandated forensic attributes.
5. Supports controlled comparative evaluation across Variants A, B, C, and D.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from phase1_leakage_free.fragment_schema_v2 import ObservableFragment
from phase2_dataset_evaluation.evaluation_engine import ReconstructedFile


def build_dual_relation_graph(
    fragment_ids: List[str],
    successor_pairs: np.ndarray,      # (M, 2) pair index array
    p_same: np.ndarray,               # Calibrated symmetric same-file probabilities
    p_next: np.ndarray,               # Calibrated forward directed probabilities
    p_prev: np.ndarray,               # Calibrated reverse directed probabilities
    embedding_sim: Optional[np.ndarray] = None,
    boundary_compat: Optional[np.ndarray] = None,
    provenance: Optional[List[Any]] = None,
    model_version: str = "2.0.0-phase3",
    top_k: int = 8,
) -> nx.DiGraph:
    """Build directed relationship graph retaining all mandated forensic attributes.

    Mandated Edge Attributes:
    1. source: source fragment identifier
    2. target: target fragment identifier
    3. p_same: symmetric same-file membership probability
    4. p_next: directed adjacency probability
    5. p_prev: reverse adjacency probability
    6. embedding_sim: deep metric embedding cosine similarity
    7. boundary_compat: byte boundary transition score
    8. provenance: origin metadata and pair indices
    9. model_version: model identifier string
    """
    g = nx.DiGraph(model_version=model_version)
    for i, fid in enumerate(fragment_ids):
        g.add_node(fid, node_index=i, fragment_id=fid)

    m_pairs = len(successor_pairs)
    if m_pairs == 0:
        return g

    if embedding_sim is None:
        embedding_sim = np.zeros(m_pairs, dtype=np.float32)
    if boundary_compat is None:
        boundary_compat = np.zeros(m_pairs, dtype=np.float32)

    # Select candidate edges combining top-k next transitions and top-k same-file candidates
    order_next = np.argsort(-p_next)
    order_same = np.argsort(-p_same)

    selected_pairs: Set[int] = set()
    kept_per_src_next: Dict[int, int] = defaultdict(int)
    for m in order_next:
        src_idx = int(successor_pairs[m, 0])
        dst_idx = int(successor_pairs[m, 1])
        if src_idx == dst_idx:
            continue
        if kept_per_src_next[src_idx] < top_k:
            kept_per_src_next[src_idx] += 1
            selected_pairs.add(m)

    kept_per_src_same: Dict[int, int] = defaultdict(int)
    for m in order_same:
        src_idx = int(successor_pairs[m, 0])
        dst_idx = int(successor_pairs[m, 1])
        if src_idx == dst_idx:
            continue
        if kept_per_src_same[src_idx] < top_k:
            kept_per_src_same[src_idx] += 1
            selected_pairs.add(m)

    for m in selected_pairs:
        src_idx = int(successor_pairs[m, 0])
        dst_idx = int(successor_pairs[m, 1])
        src_id = fragment_ids[src_idx]
        dst_id = fragment_ids[dst_idx]

        prov_val = provenance[m] if provenance is not None and m < len(provenance) else {"pair_idx": m}

        g.add_edge(
            src_id,
            dst_id,
            source=src_id,
            target=dst_id,
            p_same=float(p_same[m]),
            p_next=float(p_next[m]),
            p_prev=float(p_prev[m]),
            embedding_sim=float(embedding_sim[m]),
            boundary_compat=float(boundary_compat[m]),
            provenance=prov_val,
            model_version=model_version,
        )

    return g


class DualRelationReconstructionEngine:
    """Multi-file forensic reconstruction engine with dual-relation reasoning."""

    def __init__(
        self,
        tau_same: float = 0.50,
        tau_next: float = 0.35,
        beam_width: int = 5,
        version: str = "2.0.0-phase3",
    ):
        self.tau_same = tau_same
        self.tau_next = tau_next
        self.beam_width = beam_width
        self.version = version

    def _cluster_fragments(
        self,
        fragment_ids: List[str],
        graph: nx.DiGraph,
        tau_same: float,
    ) -> List[List[str]]:
        """Stage 1: Partition fragments into candidate file clusters using P_same."""
        # Build an undirected membership graph thresholded on symmetric P_same
        ug = nx.Graph()
        ug.add_nodes_from(fragment_ids)

        for u, v, data in graph.edges(data=True):
            p_s = data.get("p_same", 0.0)
            if p_s >= tau_same:
                ug.add_edge(u, v, weight=p_s)

        # Connected components represent candidate file clusters
        components = [list(comp) for comp in nx.connected_components(ug)]
        components.sort(key=lambda c: len(c), reverse=True)
        return components

    def _beam_search_path(
        self,
        cluster: List[str],
        graph: nx.DiGraph,
        tau_next: float,
        start_node: str,
    ) -> List[str]:
        """Stage 2: Assemble directed logical fragment path within candidate cluster."""
        cluster_set = set(cluster)
        best_path = [start_node]
        best_score = 0.0

        # Beam entry: (current_path, current_cumulative_score, visited_set)
        beam = [([start_node], 0.0, {start_node})]

        while beam:
            candidates = []
            for path, score, visited in beam:
                curr = path[-1]
                # Find outgoing edges inside the cluster
                successors = []
                for _, nxt, data in graph.out_edges(curr, data=True):
                    if nxt in cluster_set and nxt not in visited:
                        p_n = data.get("p_next", 0.0)
                        p_p = data.get("p_prev", 0.0)
                        b_compat = data.get("boundary_compat", 0.0)
                        # Transition score combines forward probability, reverse consensus, and boundary
                        transition_score = (0.7 * p_n) + (0.2 * p_p) + (0.1 * max(0.0, b_compat))
                        if p_n >= tau_next:
                            successors.append((nxt, transition_score))

                if not successors:
                    # Fallback: if no edge met tau_next, allow top candidate in cluster with highest transition score
                    fallback = []
                    for _, nxt, data in graph.out_edges(curr, data=True):
                        if nxt in cluster_set and nxt not in visited:
                            p_n = data.get("p_next", 0.0)
                            p_p = data.get("p_prev", 0.0)
                            b_compat = data.get("boundary_compat", 0.0)
                            transition_score = (0.7 * p_n) + (0.2 * p_p) + (0.1 * max(0.0, b_compat))
                            fallback.append((nxt, transition_score))
                    if fallback:
                        fallback.sort(key=lambda x: x[1], reverse=True)
                        successors = [fallback[0]]

                if not successors:
                    # End of path reached
                    if len(path) > len(best_path) or (len(path) == len(best_path) and score > best_score):
                        best_path = path
                        best_score = score
                    continue

                for nxt, t_score in successors:
                    new_path = path + [nxt]
                    new_score = score + t_score
                    new_visited = visited | {nxt}
                    candidates.append((new_path, new_score, new_visited))

            if not candidates:
                break

            # Keep top beam_width candidate paths
            candidates.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
            beam = candidates[:self.beam_width]

            # Update best path
            top_cand = beam[0]
            if len(top_cand[0]) > len(best_path) or (len(top_cand[0]) == len(best_path) and top_cand[1] > best_score):
                best_path = top_cand[0]
                best_score = top_cand[1]

        return best_path

    def reconstruct(
        self,
        graph: nx.DiGraph,
        fragment_pool: Dict[str, Union[ObservableFragment, bytes]],
        mode: str = "variant_c_dual_independent",
    ) -> List[ReconstructedFile]:
        """Execute reconstruction according to experimental variant mode.

        Modes:
            - 'variant_a_next_only': Global beam search using only P_next (v1 behavior).
            - 'variant_b_same_only': P_same clustering only, arbitrary inside-cluster order.
            - 'variant_c_dual_independent': Independent dual-model (Stage 1 P_same + Stage 2 P_next).
            - 'variant_d_multitask': Multi-task model (Stage 1 P_same + Stage 2 P_next).
        """
        all_frags = list(fragment_pool.keys())
        reconstructed_files: List[ReconstructedFile] = []
        assigned_fragments: Set[str] = set()

        def get_bytes(fid: str) -> bytes:
            item = fragment_pool[fid]
            if isinstance(item, ObservableFragment):
                return item.fragment_bytes
            return item

        if mode == "variant_a_next_only":
            # Global greedy/beam assembly without common-file clustering
            unassigned = set(all_frags)
            file_counter = 0

            # Rank candidate starting fragments by in-degree in graph
            while unassigned:
                # Pick seed with low in-degree and high out-degree
                seed = None
                best_seed_score = -1e9
                for f in unassigned:
                    in_p = sum(d.get("p_next", 0.0) for _, _, d in graph.in_edges(f, data=True) if _ in unassigned)
                    out_p = sum(d.get("p_next", 0.0) for _, _, d in graph.out_edges(f, data=True) if _ in unassigned)
                    score = out_p - in_p
                    if score > best_seed_score:
                        best_seed_score = score
                        seed = f

                if seed is None:
                    seed = next(iter(unassigned))

                # Traverse using beam search globally across all remaining unassigned
                path = self._beam_search_path(list(unassigned), graph, self.tau_next, seed)
                for f in path:
                    unassigned.discard(f)

                raw_bytes = b"".join(get_bytes(f) for f in path)
                reconstructed_files.append(ReconstructedFile(
                    recon_id=f"recon_file_{file_counter:04d}",
                    fragment_sequence=path,
                    reconstructed_bytes=raw_bytes,
                ))
                file_counter += 1

        elif mode == "variant_b_same_only":
            # Cluster strictly with P_same, but no P_next ordering inside clusters
            clusters = self._cluster_fragments(all_frags, graph, self.tau_same)
            for idx, cluster in enumerate(clusters):
                # Order arbitrarily (sorted by fragment ID or observable offset)
                ordered_seq = sorted(cluster)
                raw_bytes = b"".join(get_bytes(f) for f in ordered_seq)
                reconstructed_files.append(ReconstructedFile(
                    recon_id=f"recon_file_{idx:04d}",
                    fragment_sequence=ordered_seq,
                    reconstructed_bytes=raw_bytes,
                ))

        else:
            # Dual-relation reconstruction (Variant C & Variant D)
            # Stage 1: Partition fragments into candidate file clusters using P_same
            clusters = self._cluster_fragments(all_frags, graph, self.tau_same)

            file_counter = 0
            for cluster in clusters:
                if len(cluster) == 1:
                    raw_bytes = get_bytes(cluster[0])
                    reconstructed_files.append(ReconstructedFile(
                        recon_id=f"recon_file_{file_counter:04d}",
                        fragment_sequence=cluster,
                        reconstructed_bytes=raw_bytes,
                    ))
                    file_counter += 1
                    continue

                # Stage 2: In-cluster topological path assembly using P_next
                cluster_unassigned = set(cluster)
                while cluster_unassigned:
                    # Pick best seed within cluster
                    seed = None
                    best_seed_score = -1e9
                    for f in cluster_unassigned:
                        in_p = sum(d.get("p_next", 0.0) for _, _, d in graph.in_edges(f, data=True) if _ in cluster_unassigned)
                        out_p = sum(d.get("p_next", 0.0) for _, _, d in graph.out_edges(f, data=True) if _ in cluster_unassigned)
                        score = out_p - in_p
                        if score > best_seed_score:
                            best_seed_score = score
                            seed = f

                    if seed is None:
                        seed = next(iter(cluster_unassigned))

                    path = self._beam_search_path(list(cluster_unassigned), graph, self.tau_next, seed)
                    for f in path:
                        cluster_unassigned.discard(f)

                    raw_bytes = b"".join(get_bytes(f) for f in path)
                    reconstructed_files.append(ReconstructedFile(
                        recon_id=f"recon_file_{file_counter:04d}",
                        fragment_sequence=path,
                        reconstructed_bytes=raw_bytes,
                    ))
                    file_counter += 1

        return reconstructed_files
