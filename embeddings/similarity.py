"""Fragment embeddings and similarity (doc §13, §15)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from models.siamese import FragmentEncoder, SiameseNetwork


class FragmentEmbedder:
    """Produces learned fragment embeddings (representation learning, doc §13)."""

    name = "fragment_embedder"

    def __init__(self, input_dim: int, emb_dim: int = 32):
        self.network = SiameseNetwork(input_dim=input_dim, emb_dim=emb_dim)

    def fit(self, single_matrix, pairs, labels_same, labels_next,
            X_val_single=None, val_pairs=None, val_labels_same=None):
        self.network.fit(single_matrix, pairs, labels_same, labels_next,
                         X_val_single=X_val_single, val_pairs=val_pairs,
                         val_labels_same=val_labels_same)
        return self

    def embed(self, single_matrix: np.ndarray) -> np.ndarray:
        return self.network.embed(single_matrix)

    def save(self, path: Path) -> None:
        self.network.save(path)

    def load(self, path: Path) -> "FragmentEmbedder":
        self.network.load(path)
        return self


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pair_similarities(emb: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine similarity for each (i, j) row of ``pairs``."""
    ea, eb = emb[pairs[:, 0]], emb[pairs[:, 1]]
    num = (ea * eb).sum(axis=1)
    den = np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1) + 1e-9
    return num / den


def knn_successors(emb: np.ndarray, k: int, exclude_self: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k nearest neighbours for every fragment (candidate pruning)."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    unit = emb / norms
    sim = unit @ unit.T
    if exclude_self:
        np.fill_diagonal(sim, -1.0)
    idx = np.argpartition(-sim, kth=min(k, len(emb) - 1), axis=1)[:, :k]
    vals = np.take_along_axis(sim, idx, axis=1)
    return idx, vals


class FusedEmbedder:
    """Fuses several embedding spaces (handcrafted-MLP + raw-byte CNN).

    Candidate successor generation takes the union of per-space nearest
    neighbours; pair similarity is the mean cosine across spaces (doc §13-14,
    deep + metric learning over both representations).
    """

    def __init__(self, embedders: dict):
        self.embedders = embedders   # name -> object with .embed(matrix)

    def embed_all(self, matrices: dict) -> dict:
        """``matrices``: name -> fragment matrix matching each embedder input."""
        return {name: self.embedders[name].embed(matrices[name])
                for name in self.embedders if name in matrices}

    @staticmethod
    def knn_union(embeddings: dict, k: int, cap: int = 14):
        """Union of top-k neighbours across all spaces, capped at ``cap``."""
        n = len(next(iter(embeddings.values())))
        k_eff = min(k, n - 1)
        sets = [set() for _ in range(n)]
        for emb in embeddings.values():
            idx, vals = knn_successors(emb, k_eff)
            for i in range(n):
                for j, v in zip(idx[i], vals[i]):
                    if j != i:
                        sets[i].add((int(j), float(v)))
        out_idx = np.zeros((n, cap), dtype=np.int64)
        out_val = np.zeros((n, cap))
        for i in range(n):
            top = sorted(sets[i], key=lambda t: -t[1])[:cap]
            for j, (jj, vv) in enumerate(top):
                out_idx[i, j], out_val[i, j] = jj, vv
        return out_idx, out_val

    @staticmethod
    def pair_similarity(embeddings: dict, pairs: np.ndarray) -> np.ndarray:
        sims = None
        for emb in embeddings.values():
            s = pair_similarities(emb, pairs)
            sims = s if sims is None else sims + s
        return sims / max(1, len(embeddings))
