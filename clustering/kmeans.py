"""Unsupervised clustering of fragment embeddings (doc §16): K-Means, DBSCAN
and hierarchical clustering as auxiliary inference and analysis mechanisms.
Clusters are treated as fragment families, never as ground-truth files."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score


class KMeansClusterer:
    name = "kmeans"

    def __init__(self, k_range: Tuple[int, int] = (4, 14), seed: int = 42):
        self.k_range = k_range
        self.seed = seed
        self.model: Optional[KMeans] = None
        self.k_ = 0
        self.silhouette_ = 0.0

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        best, best_k, best_s = None, self.k_range[0], -1.0
        for k in range(self.k_range[0], min(self.k_range[1], max(2, len(X) - 1)) + 1):
            m = KMeans(n_clusters=k, n_init=10, random_state=self.seed)
            labels = m.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            s = float(silhouette_score(X, labels, sample_size=min(2000, len(X)),
                                       random_state=self.seed))
            if s > best_s:
                best, best_k, best_s = m, k, s
        self.model, self.k_, self.silhouette_ = best, best_k, best_s
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(X).astype(int)


class DBSCANClusterer:
    name = "dbscan"

    def __init__(self, min_samples: int = 5, eps: Optional[float] = None):
        self.min_samples = min_samples
        self.eps = eps
        self.model: Optional[DBSCAN] = None
        self.n_clusters_ = 0

    def fit(self, X: np.ndarray) -> "DBSCANClusterer":
        eps = self.eps
        if eps is None:
            # heuristic: median distance to 5th nearest neighbour
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=self.min_samples).fit(X)
            dist, _ = nn.kneighbors(X)
            eps = float(np.median(dist[:, -1])) or 0.1
        self.model = DBSCAN(eps=eps, min_samples=self.min_samples)
        labels = self.model.fit_predict(X)
        self.n_clusters_ = len(set(labels) - {-1})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.fit_predict(X).astype(int)


class HierarchicalClusterer:
    name = "hierarchical"

    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.model = AgglomerativeClustering(n_clusters=n_clusters)

    def fit(self, X: np.ndarray) -> "HierarchicalClusterer":
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(X).astype(int)


def cluster_summary(labels: np.ndarray, metas: list) -> Dict[str, object]:
    """Per-cluster size and dominant file-type composition."""
    out: Dict[str, object] = {}
    for c in sorted(set(labels.tolist())):
        member_idx = np.where(labels == c)[0]
        types: Dict[str, int] = {}
        for i in member_idx:
            t = str(metas[i].get("file_type") or "unknown")
            types[t] = types.get(t, 0) + 1
        out[str(int(c))] = {"size": int(len(member_idx)),
                            "composition": dict(sorted(types.items(), key=lambda kv: -kv[1]))}
    return out
