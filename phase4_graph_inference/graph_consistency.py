"""Graph Consistency Checker and Conflict Resolution Engine (v2.0.0-phase4).

Detects, quantifies, and repairs topological and forensic graph inconsistencies:
1. Directed cycles (simple cycles violating linear file topology).
2. Multiple predecessors (in-degree ambiguity: which fragment came before v?).
3. Multiple successors (out-degree ambiguity: which fragment follows u?).
4. Contradictory predictions: P_next(u, v) high while P_same(u, v) low.
5. Transitivity violations in same-file equivalence relations.
6. Feedback Arc Set cycle repair.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


@dataclass
class InconsistencyReport:
    """Detailed report on topological and forensic graph anomalies."""
    cycle_count: int
    cycle_samples: List[List[str]]
    in_degree_conflicts: int
    out_degree_conflicts: int
    contradictory_edges: int
    transitivity_violations: int
    conflict_entropy: float
    is_topologically_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_count": self.cycle_count,
            "cycle_samples": self.cycle_samples[:5],
            "in_degree_conflicts": self.in_degree_conflicts,
            "out_degree_conflicts": self.out_degree_conflicts,
            "contradictory_edges": self.contradictory_edges,
            "transitivity_violations": self.transitivity_violations,
            "conflict_entropy": round(self.conflict_entropy, 4),
            "is_topologically_valid": self.is_topologically_valid,
        }


class GraphConsistencyChecker:
    """Detects and repairs topological and relationship contradictions."""

    def __init__(self, tau_conflict: float = 0.20, tau_contradiction_margin: float = 0.30):
        self.tau_conflict = tau_conflict
        self.tau_contradiction_margin = tau_contradiction_margin

    def check_consistency(self, graph: nx.DiGraph) -> InconsistencyReport:
        """Run complete consistency audit across the directed candidate graph."""
        # 1. Directed cycles (scalable bounded sampling to avoid exponential combinatorial explosion)
        is_dag = nx.is_directed_acyclic_graph(graph)
        if is_dag:
            cycle_count = 0
            cycles = []
        else:
            import itertools
            cycles = list(itertools.islice(nx.simple_cycles(graph), 25))
            scc_count = sum(1 for c in nx.strongly_connected_components(graph) if len(c) > 1)
            cycle_count = max(len(cycles), scc_count)

        # 2. Branching conflicts (multiple high-confidence predecessors / successors)
        in_conflicts = 0
        out_conflicts = 0
        branching_entropies = []

        for node in graph.nodes():
            in_edges = graph.in_edges(node, data=True)
            high_in = [d.get("p_next", 0.0) for _, _, d in in_edges if d.get("p_next", 0.0) >= self.tau_conflict]
            if len(high_in) > 1:
                in_conflicts += 1
                probs = np.array(high_in) / (sum(high_in) + 1e-12)
                branching_entropies.append(-np.sum(probs * np.log2(probs + 1e-12)))

            out_edges = graph.out_edges(node, data=True)
            high_out = [d.get("p_next", 0.0) for _, _, d in out_edges if d.get("p_next", 0.0) >= self.tau_conflict]
            if len(high_out) > 1:
                out_conflicts += 1
                probs = np.array(high_out) / (sum(high_out) + 1e-12)
                branching_entropies.append(-np.sum(probs * np.log2(probs + 1e-12)))

        mean_conflict_entropy = float(np.mean(branching_entropies)) if branching_entropies else 0.0

        # 3. Contradictory edges: high P_next but low P_same
        contradictory_count = 0
        for u, v, data in graph.edges(data=True):
            p_n = data.get("p_next", 0.0)
            p_s = data.get("p_same", 0.0)
            # If P_next is confident transition but P_same claims totally different files
            if p_n >= 0.35 and (p_n - p_s) >= self.tau_contradiction_margin:
                contradictory_count += 1

        # 4. Transitivity violations in same-file subgraphs (sample-based check)
        transitivity_violations = 0
        ug = nx.Graph()
        for u, v, data in graph.edges(data=True):
            if data.get("p_same", 0.0) >= 0.40:
                ug.add_edge(u, v)

        # In an ideal same-file relation, connected components should be cliques
        for comp in nx.connected_components(ug):
            if 3 <= len(comp) <= 20:
                sub = ug.subgraph(comp)
                possible_edges = len(comp) * (len(comp) - 1) // 2
                actual_edges = sub.number_of_edges()
                transitivity_violations += max(0, possible_edges - actual_edges)

        is_valid = (cycle_count == 0 and in_conflicts == 0 and out_conflicts == 0 and contradictory_count == 0)

        return InconsistencyReport(
            cycle_count=cycle_count,
            cycle_samples=cycles[:5],
            in_degree_conflicts=in_conflicts,
            out_degree_conflicts=out_conflicts,
            contradictory_edges=contradictory_count,
            transitivity_violations=transitivity_violations,
            conflict_entropy=mean_conflict_entropy,
            is_topologically_valid=is_valid,
        )

    def repair_feedback_arc_set(self, graph: nx.DiGraph, max_removals: int = 150) -> nx.DiGraph:
        """Break directed cycles using bounded heuristic feedback arc set removal."""
        repaired = graph.copy()
        removals = 0
        while not nx.is_directed_acyclic_graph(repaired) and removals < max_removals:
            try:
                cycle = nx.find_cycle(repaired, orientation="original")
                # Identify edge in cycle with smallest transition evidence
                min_edge = None
                min_weight = 1e9
                for u, v, _ in cycle:
                    w = repaired[u][v].get("p_next", 0.0)
                    if w < min_weight:
                        min_weight = w
                        min_edge = (u, v)

                if min_edge is not None:
                    repaired.remove_edge(*min_edge)
                    removals += 1
                else:
                    break
            except nx.NetworkXNoCycle:
                break
        return repaired
