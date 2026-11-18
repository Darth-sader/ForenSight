"""Fragment relationship graph (doc §18): fragments as nodes, ML predictions
as weighted directed edges combining supervised probabilities, embedding
similarity, structural compatibility and contextual information."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def build_relationship_graph(fragment_ids: List[str],
                             successor_pairs: np.ndarray,        # (m, 2) row indices
                             p_next: np.ndarray,                 # calibrated P(B follows A)
                             p_same: np.ndarray,
                             emb_sim: np.ndarray,
                             structural: np.ndarray,
                             context: np.ndarray,
                             anomaly: np.ndarray,
                             top_k: int = 5) -> nx.DiGraph:
    """Directed graph keeping the top-k highest-scoring successors per fragment."""
    g = nx.DiGraph()
    for i, fid in enumerate(fragment_ids):
        g.add_node(i, fragment_id=fid, anomaly=float(anomaly[i]))
    order = np.argsort(-p_next)
    kept_per_src: Dict[int, int] = {}
    for m in order:
        a, b = int(successor_pairs[m, 0]), int(successor_pairs[m, 1])
        if a == b:
            continue
        if kept_per_src.get(a, 0) >= top_k:
            continue
        kept_per_src[a] = kept_per_src.get(a, 0) + 1
        g.add_edge(a, b,
                   p_next=float(p_next[m]),
                   p_same=float(p_same[m]),
                   emb_sim=float(emb_sim[m]),
                   structural=float(structural[m]),
                   context=float(context[m]))
    return g


def edge_combined_score(attrs: Dict[str, float], weights: Optional[Dict[str, float]] = None,
                        anomaly_b: float = 0.0) -> float:
    """Combined transition evidence for one edge (doc §20 scoring components)."""
    w = weights or {"transition": 0.55, "embedding": 0.2, "structural": 0.15,
                    "context": 0.1, "anomaly_penalty": 0.1}
    s = (w["transition"] * attrs["p_next"] +
         w["embedding"] * max(0.0, attrs["emb_sim"]) +
         w["structural"] * attrs["structural"] +
         w["context"] * attrs["context"] -
         w["anomaly_penalty"] * anomaly_b)
    return float(s)


def graph_stats(g: nx.DiGraph) -> Dict[str, object]:
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "avg_out_degree": float(np.mean([d for _, d in g.out_degree()])) if g.number_of_nodes() else 0.0,
        "density": nx.density(g),
        "strongly_connected": nx.number_strongly_connected_components(g),
    }
