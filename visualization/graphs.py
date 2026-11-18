"""Fragment relationship graph visualisation (doc §37)."""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from config import PLOTS_DIR


def plot_relationship_graph(g: nx.DiGraph, fragment_id_of=None,
                            anomaly: Optional[Dict[int, float]] = None,
                            name: str = "relationship_graph",
                            max_nodes: int = 160, seed: int = 42) -> str:
    """Spring layout of the relationship graph, edges shaded by transition
    probability, nodes shaded by anomaly score."""
    if g.number_of_nodes() > max_nodes:
        keep = sorted(g.degree, key=lambda kv: -kv[1])[:max_nodes]
        sub = g.subgraph([n for n, _ in keep]).copy()
    else:
        sub = g
    pos = nx.spring_layout(sub, seed=seed, k=1.2 / max(2, np.sqrt(sub.number_of_nodes())))
    fig, ax = plt.subplots(figsize=(11, 9))

    edge_scores = [d.get("p_next", 0.0) for _, _, d in sub.edges(data=True)]
    node_anom = [anomaly.get(n, 0.0) if anomaly else 0.0 for n in sub.nodes()]
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color=edge_scores,
                           edge_cmap=plt.cm.viridis, edge_vmin=0, vmax=1,
                           width=0.7, alpha=0.6, arrows=True, arrowsize=4)
    nc = ax.scatter([], [])
    nodes = nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=node_anom,
                                   cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1,
                                   node_size=28, linewidths=0.2, edgecolors="black")
    labels = {n: fragment_id_of(n) if fragment_id_of else str(n)
              for n in list(sub.nodes())[:0]}  # labels off by default (clutter)
    nx.draw_networkx_labels(sub, pos, labels, ax=ax, font_size=4)
    ax.set_title(f"Fragment relationship graph ({sub.number_of_nodes()} nodes, "
                 f"{sub.number_of_edges()} ML-scored edges)")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb1.set_label("P(B follows A)")
    sm2 = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax, fraction=0.03, pad=0.06)
    cb2.set_label("Anomaly score")
    ax.axis("off")
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def plot_candidate_path(g: nx.DiGraph, sequence: List[int], fragment_id_of,
                        name: str = "candidate_path") -> str:
    """Highlight one candidate reconstruction path inside the graph."""
    path_graph = g.subgraph(sequence).copy()
    pos = {n: (i, 0) for i, n in enumerate(sequence)}
    fig, ax = plt.subplots(figsize=(max(6, len(sequence) * 0.7), 2.4))
    scores = [path_graph.edges[sequence[i], sequence[i + 1]].get("p_next", 0)
              for i in range(len(sequence) - 1)]
    nx.draw_networkx_edges(path_graph, pos, ax=ax, edge_color=scores,
                           edge_cmap=plt.cm.viridis, edge_vmin=0, vmax=1,
                           width=2, arrows=True, arrowsize=12)
    nx.draw_networkx_nodes(path_graph, pos, ax=ax, node_color="#4c72b0",
                           node_size=380, edgecolors="black")
    nx.draw_networkx_labels(path_graph, pos,
                            {n: fragment_id_of(n) for n in sequence},
                            ax=ax, font_size=6)
    for i, s in enumerate(scores):
        ax.text(i + 0.5, 0.13, f"{s:.2f}", ha="center", fontsize=6, color="#333")
    ax.set_title("Candidate reconstruction path (edge labels: calibrated transition probability)")
    ax.axis("off")
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def graph_to_frontend(g: nx.DiGraph, fragment_id_of=None, anomaly=None,
                      max_nodes: int = 400, max_edges: int = 2500) -> Dict[str, object]:
    """Serialize (a pruned copy of) the graph as JSON for the dashboard."""
    if g.number_of_nodes() > max_nodes:
        keep = sorted(g.degree, key=lambda kv: -kv[1])[:max_nodes]
        sub = g.subgraph([n for n, _ in keep]).copy()
    else:
        sub = g
    nodes = [{"id": fragment_id_of(n) if fragment_id_of else str(n),
              "index": int(n),
              "anomaly": round(float(anomaly.get(n, 0.0)), 3) if anomaly else 0.0,
              "degree": int(sub.degree(n))}
             for n in sub.nodes()]
    node_ids = {n: i for i, n in enumerate(sub.nodes())}
    edges = [{"source": node_ids[a], "target": node_ids[b],
              "p_next": round(float(d.get("p_next", 0.0)), 3),
              "emb_sim": round(float(d.get("emb_sim", 0.0)), 3)}
             for a, b, d in sub.edges(data=True)][:max_edges]
    return {"nodes": nodes, "edges": edges}
