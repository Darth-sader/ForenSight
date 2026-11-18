"""Path optimisation (doc §19-20): find high-scoring *valid* reconstructions.

Combines beam-search candidates from :mod:`graph_inference` with forensic
constraint checking and the composite reconstruction score:

    FinalScore = ML transition + embedding similarity + structural consistency
                 + contextual evidence - anomaly penalty - violations
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np

from config import SCORE_WEIGHTS
from graph.graph_inference import CandidatePath, beam_search
from graph.relationship_graph import edge_combined_score


def optimize_from_seeds(
    graph: nx.DiGraph,
    seeds: List[int],
    constraints_fn: Callable[[CandidatePath], tuple],   # -> (violations:int, issues:list)
    beam_width: int = 6,
    weights: Optional[Dict[str, float]] = None,
    forbidden: Optional[set] = None,
    max_candidates_per_seed: int = 5,
) -> List[dict]:
    """Run beam search from every seed, score candidates and return ranked
    hypothesis dicts (unsorted across seeds - global ranking is the ranker's job)."""
    hypotheses: List[dict] = []
    for seed in seeds:
        paths = beam_search(graph, seed, beam_width=beam_width, weights=weights,
                            forbidden=forbidden)
        for rank, path in enumerate(paths[: max(1, max_candidates_per_seed)]):
            violations, issues = constraints_fn(path)
            components = path_components(path)
            raw = combined_score(components, violations)
            hypotheses.append({
                "seed": seed, "rank_in_seed": rank, "path": path,
                "violations": violations, "issues": issues,
                "components": components, "raw_score": float(raw),
            })
    return hypotheses


def path_components(path: CandidatePath) -> Dict[str, float]:
    """Component evidence of a candidate path (doc §20)."""
    if not path.edge_attrs:
        return {"transition": 0.0, "embedding": 0.0, "structural": 0.0,
                "context": 0.0, "anomaly": 0.0, "gap_ratio": 0.0}
    transition = float(np.mean([a["p_next"] for a in path.edge_attrs]))
    embedding = float(np.mean([max(0.0, a["emb_sim"]) for a in path.edge_attrs]))
    structural = float(np.mean([a["structural"] for a in path.edge_attrs]))
    context = float(np.mean([a["context"] for a in path.edge_attrs]))
    anomaly = 0.0
    return {"transition": transition, "embedding": embedding, "structural": structural,
            "context": context, "anomaly": anomaly,
            "gap_ratio": len(path.gaps) / max(1, len(path.nodes) - 1)}


def combined_score(components: Dict[str, float], violations: int,
                   weights: Optional[Dict[str, float]] = None) -> float:
    w = weights or SCORE_WEIGHTS
    score = (w["transition"] * components["transition"] +
             w["embedding"] * components["embedding"] +
             w["structural"] * components["structural"] +
             w["context"] * components["context"] -
             w["anomaly_penalty"] * components.get("anomaly", 0.0) -
             w["violation_penalty"] * violations)
    return float(np.clip(score, 0.0, 1.0))


def greedy_baseline_path(graph: nx.DiGraph, seed: int, max_len: int = 64) -> Optional[List[int]]:
    """Highest-probability edge walk (baseline that the beam search improves on)."""
    path, seen = [seed], {seed}
    cur = seed
    for _ in range(max_len):
        best, best_s = None, -1.0
        for _, nxt, attrs in graph.out_edges(cur, data=True):
            if nxt in seen:
                continue
            s = attrs["p_next"]
            if s > best_s:
                best, best_s = nxt, s
        if best is None:
            break
        path.append(best)
        seen.add(best)
        cur = best
    return path if len(path) > 1 else None
