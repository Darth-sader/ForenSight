"""Forensic Graph Diagnostics and Structural Auditing (v2.0.0-phase4).

Computes comprehensive forensic topological metrics on candidate graphs:
1. Connected components and clustering coefficients.
2. In-degree and out-degree distributions and entropy.
3. Strongly connected components and acyclicity checks.
4. Reachability from candidate seed fragments.
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
class GraphDiagnosticSummary:
    """Summary of global topological metrics for a forensic relationship graph."""
    num_nodes: int
    num_edges: int
    density: float
    num_weakly_connected: int
    num_strongly_connected: int
    num_isolated: int
    avg_clustering_coeff: float
    mean_in_degree: float
    max_in_degree: int
    mean_out_degree: float
    max_out_degree: int
    is_dag: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": round(self.density, 6),
            "num_weakly_connected_components": self.num_weakly_connected,
            "num_strongly_connected_components": self.num_strongly_connected,
            "num_isolated_nodes": self.num_isolated,
            "avg_clustering_coefficient": round(self.avg_clustering_coeff, 4),
            "mean_in_degree": round(self.mean_in_degree, 4),
            "max_in_degree": self.max_in_degree,
            "mean_out_degree": round(self.mean_out_degree, 4),
            "max_out_degree": self.max_out_degree,
            "is_directed_acyclic_graph": self.is_dag,
        }


class GraphDiagnostics:
    """Computes forensic graph health and diagnostic metrics."""

    @staticmethod
    def diagnose_graph(graph: nx.DiGraph) -> GraphDiagnosticSummary:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()

        if n == 0:
            return GraphDiagnosticSummary(
                num_nodes=0, num_edges=0, density=0.0, num_weakly_connected=0,
                num_strongly_connected=0, num_isolated=0, avg_clustering_coeff=0.0,
                mean_in_degree=0.0, max_in_degree=0, mean_out_degree=0.0, max_out_degree=0,
                is_dag=True,
            )

        density = nx.density(graph)
        wcc = nx.number_weakly_connected_components(graph)
        scc = nx.number_strongly_connected_components(graph)
        isolated = sum(1 for _, d in graph.degree() if d == 0)

        in_degs = [d for _, d in graph.in_degree()]
        out_degs = [d for _, d in graph.out_degree()]

        mean_in = float(np.mean(in_degs)) if in_degs else 0.0
        max_in = max(in_degs) if in_degs else 0
        mean_out = float(np.mean(out_degs)) if out_degs else 0.0
        max_out = max(out_degs) if out_degs else 0

        # Undirected view for clustering
        ug = graph.to_undirected()
        avg_clustering = float(nx.average_clustering(ug)) if n > 1 else 0.0
        is_dag = nx.is_directed_acyclic_graph(graph)

        return GraphDiagnosticSummary(
            num_nodes=n,
            num_edges=m,
            density=density,
            num_weakly_connected=wcc,
            num_strongly_connected=scc,
            num_isolated=isolated,
            avg_clustering_coeff=avg_clustering,
            mean_in_degree=mean_in,
            max_in_degree=max_in,
            mean_out_degree=mean_out,
            max_out_degree=max_out,
            is_dag=is_dag,
        )
