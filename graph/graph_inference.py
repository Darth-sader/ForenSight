"""Graph-based inference (doc §19): generate multiple candidate reconstruction
paths via beam search over the fragment relationship graph.

Deliberately avoids choosing the highest edge at every step: the beam keeps
the ``beam_width`` best partial paths at every expansion so competing
hypotheses survive (doc §19, §32).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from config import (BEAM_WIDTH, GAP_CONFIDENCE_THRESHOLD, MAX_CANDIDATES_PER_SEED,
                    MAX_SEQUENCE_LENGTH, SCORE_WEIGHTS)
from graph.relationship_graph import edge_combined_score

GAP = "[UNKNOWN]"


@dataclass
class CandidatePath:
    """One candidate fragment sequence produced by inference."""
    seed: int
    nodes: List[int]
    gaps: List[int]                       # indices in `nodes` after which a gap was inferred
    edge_scores: List[float]
    edge_attrs: List[Dict[str, float]] = field(default_factory=list)
    direction: str = "forward"            # forward | backward | stitched

    @property
    def avg_score(self) -> float:
        return float(np.mean(self.edge_scores)) if self.edge_scores else 0.0

    @property
    def min_score(self) -> float:
        return float(np.min(self.edge_scores)) if self.edge_scores else 0.0

    def sequence_ids(self, id_of) -> List[str]:
        return [id_of(n) for n in self.nodes]


def _step(graph: nx.DiGraph, cand: CandidatePath, forward: bool,
          forbidden: set, weights: Optional[Dict[str, float]],
          gap_threshold: float, gap_hysteresis: float,
          allowed: Optional[set]) -> List[CandidatePath]:
    """One expansion step of the beam in either direction.

    forward: extend with the best successor of the last node;
    backward: prepend the best predecessor of the first node.
    Returns the new partial paths ([] when the walk cannot continue).
    """
    anchor = cand.nodes[-1] if forward else cand.nodes[0]
    best = None
    if forward:
        candidates = graph.out_edges(anchor, data=True)
    else:
        candidates = graph.in_edges(anchor, data=True)
    for u, v, attrs in sorted(candidates,
                              key=lambda e: -edge_combined_score(e[2], weights,
                                                                 float(graph.nodes[e[1] if forward else e[0]].get("anomaly", 0.0)))):
        nxt = v if forward else u
        if nxt in cand.nodes or nxt in forbidden:
            continue
        if allowed is not None and nxt not in allowed:
            continue
        best = (nxt, attrs)
        break
    if best is None:
        return []
    nxt, attrs = best
    anomaly_b = float(graph.nodes[nxt].get("anomaly", 0.0))
    score = edge_combined_score(attrs, weights, anomaly_b)
    if forward:
        nodes = cand.nodes + [nxt]
        gap_at = len(cand.nodes) - 1
    else:
        nodes = [nxt] + cand.nodes
        gap_at = 0
    ncand = CandidatePath(seed=cand.seed, nodes=nodes,
                          gaps=list(cand.gaps),
                          edge_scores=cand.edge_scores + [score],
                          edge_attrs=cand.edge_attrs + [dict(attrs)],
                          direction=cand.direction)
    out = [ncand]
    if score < gap_threshold and not ncand.gaps:
        ncand.gaps.append(gap_at)
        ncand.edge_scores[-1] = min(score, gap_threshold) * 0.8
    elif gap_threshold <= score < gap_threshold + gap_hysteresis and not cand.gaps:
        g = CandidatePath(seed=cand.seed, nodes=list(nodes),
                          gaps=sorted(cand.gaps + [gap_at]),
                          edge_scores=cand.edge_scores[:-1] + [score * 0.85],
                          edge_attrs=cand.edge_attrs + [dict(attrs)],
                          direction=cand.direction)
        out.append(g)
    return out


def beam_search(graph: nx.DiGraph, seed: int, beam_width: int = BEAM_WIDTH,
                max_len: int = MAX_SEQUENCE_LENGTH,
                gap_threshold: float = GAP_CONFIDENCE_THRESHOLD,
                weights: Optional[Dict[str, float]] = None,
                forbidden: Optional[set] = None,
                min_continuation: float = 0.08,
                allowed: Optional[set] = None,
                direction: str = "forward") -> List[CandidatePath]:
    """Return the top candidate paths starting at ``seed``.

    Bidirectional support (doc §19 extension): ``direction="backward"`` walks
    the reversed graph from a footer seed.  When the best continuation edge of
    a partial path falls below ``gap_threshold`` a [UNKNOWN] gap is emitted and
    the walk continues at reduced score (doc §31: never fabricate fragments).
    """
    beam: List[CandidatePath] = [CandidatePath(seed=seed, nodes=[seed], gaps=[],
                                               edge_scores=[], direction=direction)]
    finished: List[CandidatePath] = []
    forbidden = forbidden or set()
    gap_hysteresis = 0.12

    for _ in range(max_len):
        expansions: List[CandidatePath] = []
        for cand in beam:
            expansions.extend(_step(graph, cand, direction == "forward", forbidden,
                                    weights, gap_threshold, gap_hysteresis, allowed))
        if not expansions:
            break
        expansions.sort(key=lambda c: -_path_rank(c))
        beam = expansions[:beam_width]
        finished.extend(beam)
    if not finished:
        finished = beam
    # dedup by node sequence, rank, keep best per distinct sequence
    seen: Dict[Tuple[int, ...], CandidatePath] = {}
    for c in sorted(finished, key=lambda c: -_path_rank(c)):
        seen.setdefault(tuple(c.nodes), c)
    out = list(seen.values())
    out.sort(key=lambda c: -_path_rank(c))
    return out[:MAX_CANDIDATES_PER_SEED]


def backward_beam_search(graph: nx.DiGraph, seed: int, **kwargs) -> List[CandidatePath]:
    """Bidirectional reconstruction: walk predecessors from a footer seed."""
    return beam_search(graph, seed, direction="backward", **kwargs)


def island_partition(graph: nx.DiGraph, p_threshold: float = 0.5) -> List[set]:
    """Island detection (doc §16 + §19 extension): connected components of the
    high-confidence subgraph (edges with p_next >= threshold).  Reconstruction
    runs per island, which shrinks the search space and keeps cross-file
    contamination from bridging unrelated fragments."""
    strong = nx.DiGraph()
    strong.add_nodes_from(graph.nodes())
    for u, v, d in graph.edges(data=True):
        if d.get("p_next", 0.0) >= p_threshold:
            strong.add_edge(u, v)
    return [set(c) for c in nx.weakly_connected_components(strong)
            if len(c) > 1]


def stitch_bidirectional(forward_paths: List[CandidatePath],
                         backward_paths: List[CandidatePath]) -> List[CandidatePath]:
    """Stitch forward paths (from headers) with backward paths (from footers)
    that meet at a shared fragment, producing full-sequence candidates."""
    by_start: Dict[int, CandidatePath] = {}
    for b in backward_paths:
        by_start.setdefault(b.nodes[0], b)
    stitched: List[CandidatePath] = []
    for f in forward_paths:
        b = by_start.get(f.nodes[-1])
        if b is None:
            continue
        nodes = f.nodes + b.nodes[1:]
        if len(set(nodes)) != len(nodes):
            continue
        gaps = f.gaps + [g + len(f.nodes) - 1 for g in b.gaps]
        stitched.append(CandidatePath(
            seed=f.seed, nodes=nodes, gaps=gaps,
            edge_scores=f.edge_scores + b.edge_scores,
            edge_attrs=f.edge_attrs + b.edge_attrs,
            direction="stitched"))
    return stitched


def _path_rank(c: CandidatePath) -> float:
    if not c.edge_scores:
        return 0.0
    return float(np.mean(c.edge_scores) * 0.8 + np.min(c.edge_scores) * 0.2)
