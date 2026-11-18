"""Visualisation of fragment embedding space (doc §15, §37): PCA and t-SNE
projections of learned embeddings, optionally coloured by file type or
cluster."""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR

TYPE_COLORS = {"JPEG": "#d62728", "PNG": "#1f77b4", "PDF": "#2ca02c", "ZIP": "#9467bd",
               "DOCX": "#ff7f0e", "XLSX": "#8c564b", "MP3": "#17becf", "MP4": "#e377c2",
               "unknown": "#7f7f7f"}


def _scatter(ax, xy, labels, label_names, title):
    for lab in sorted(set(labels.tolist())):
        m = labels == lab
        name = label_names.get(lab, str(lab)) if label_names else str(lab)
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.6,
                   color=TYPE_COLORS.get(name, "#7f7f7f"), label=name)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_embeddings(emb: np.ndarray, file_types: List[str],
                    clusters: Optional[np.ndarray] = None,
                    method: str = "tsne", name: str = "embeddings",
                    max_points: int = 2500, seed: int = 42) -> Dict[str, str]:
    """Save PCA + t-SNE (or requested method) projections as PNG."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(emb), min(max_points, len(emb)), replace=False) \
        if len(emb) > max_points else np.arange(len(emb))
    e, ft = emb[idx], [file_types[i] for i in idx]
    paths: Dict[str, str] = {}

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(e)
    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(ax, xy, np.array(ft), {t: t for t in set(ft)},
             f"Fragment embeddings - PCA ({len(e)} fragments)")
    ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}_pca.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths["pca"] = str(p)

    if method in ("tsne", "both"):
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, perplexity=min(30, max(5, len(e) // 50)),
                  random_state=seed, init="pca").fit_transform(e)
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter(ax, xy, np.array(ft), {t: t for t in set(ft)},
                 f"Fragment embeddings - t-SNE ({len(e)} fragments)")
        ax.legend(markerscale=2, fontsize=7, loc="best")
        fig.tight_layout()
        p = PLOTS_DIR / f"{name}_tsne.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths["tsne"] = str(p)

    if clusters is not None:
        cl = clusters[idx]
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter(ax, xy, cl, None, "Fragment embeddings - clusters (t-SNE)")
        fig.tight_layout()
        p = PLOTS_DIR / f"{name}_clusters.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths["clusters"] = str(p)
    return paths


def plot_cluster_composition(cluster_summary: Dict[str, object],
                             name: str = "cluster_composition") -> str:
    """Stacked bar of file-type composition per cluster."""
    fig, ax = plt.subplots(figsize=(9, 5))
    clusters = sorted(cluster_summary.keys(), key=lambda k: int(k))
    all_types: List[str] = []
    for c in clusters:
        for t in cluster_summary[c]["composition"]:
            if t not in all_types:
                all_types.append(t)
    bottom = np.zeros(len(clusters))
    for t in all_types:
        vals = np.array([cluster_summary[c]["composition"].get(t, 0) for c in clusters],
                        dtype=float)
        ax.bar(clusters, vals, bottom=bottom,
               color=TYPE_COLORS.get(t, "#7f7f7f"), label=t)
        bottom += vals
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Fragments")
    ax.set_title("Cluster composition by file type")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)
