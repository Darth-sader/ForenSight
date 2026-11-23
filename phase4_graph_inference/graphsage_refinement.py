"""GraphSAGE Relational Graph Refinement Network (v2.0.0-phase4).

Refines pairwise relationship probabilities by aggregating structural graph context
across fragment neighborhoods:
1. Fragment node representation h_v in R^{d_node}
2. Edge feature representation e_{uv} in R^{d_edge}
3. Neighborhood Mean Aggregator:
       h_{N(v)} = Mean_{u in N(v)}(h_u)
       h_v^{(l+1)} = ReLU(BatchNorm(W_self * h_v^{(l)} + W_neigh * h_{N(v)}))
4. Relational Edge MLP:
       [h_u^{(L)} || h_v^{(L)} || e_{uv}] -> (P_next_refined, P_same_refined)
5. Strictly confines graph message passing within partition boundaries.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer with mean aggregation."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.linear_self = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node feature matrix
            adj_norm: (N, N) row-normalized adjacency matrix
        """
        neigh = torch.mm(adj_norm, x)
        h = self.linear_self(x) + self.linear_neigh(neigh)
        h = self.norm(h)
        h = self.act(h)
        return self.dropout(h)


class GraphSAGENet(nn.Module):
    """2-Layer GraphSAGE network with relational edge refinement heads."""

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        edge_dim: int = 6,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # Input projection for high-dimensional raw single features (e.g. 400 dims)
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GraphSAGE message-passing layers
        self.sage1 = GraphSAGELayer(hidden_dim, hidden_dim, dropout=dropout)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim, dropout=dropout)

        # Edge refinement MLP: [h_u || h_v || e_uv]
        edge_input_dim = (hidden_dim * 2) + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # (logit_next, logit_same)
        )

    def forward(
        self,
        node_features: torch.Tensor,     # (N, node_dim)
        adj_norm: torch.Tensor,          # (N, N)
        edge_indices: torch.Tensor,      # (E, 2)
        edge_features: torch.Tensor,     # (E, edge_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute refined next-fragment and same-file probabilities for candidate edges."""
        # 1. Project node features
        h0 = self.node_proj(node_features)

        # 2. Message passing
        h1 = self.sage1(h0, adj_norm)
        h2 = self.sage2(h1, adj_norm)

        # 3. Form edge representations
        u_idx = edge_indices[:, 0]
        v_idx = edge_indices[:, 1]
        h_u = h2[u_idx]
        h_v = h2[v_idx]

        edge_cat = torch.cat([h_u, h_v, edge_features], dim=-1)
        logits = self.edge_mlp(edge_cat)

        logit_next = logits[:, 0]
        logit_same = logits[:, 1]

        p_next = torch.sigmoid(logit_next)
        p_same = torch.sigmoid(logit_same)

        return p_next, p_same


class GraphSAGERefiner:
    """Trainer and inference interface for GraphSAGE edge refinement."""

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        edge_dim: int = 6,
        lr: float = 2e-3,
        epochs: int = 40,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = GraphSAGENet(node_dim=node_dim, hidden_dim=hidden_dim, edge_dim=edge_dim).to(self.device)
        self.lr = lr
        self.epochs = epochs

    def _normalize_adj(self, edge_indices: np.ndarray, n_nodes: int) -> torch.Tensor:
        """Construct row-normalized adjacency matrix with self-loops."""
        adj = np.eye(n_nodes, dtype=np.float32)
        if len(edge_indices) > 0:
            adj[edge_indices[:, 0], edge_indices[:, 1]] = 1.0
        row_sums = np.sum(adj, axis=1, keepdims=True) + 1e-12
        adj_norm = adj / row_sums
        return torch.tensor(adj_norm, dtype=torch.float32, device=self.device)

    def fit(
        self,
        node_features: np.ndarray,      # (N_train, node_dim)
        edge_indices: np.ndarray,       # (E_train, 2)
        edge_features: np.ndarray,      # (E_train, edge_dim)
        y_next: np.ndarray,             # (E_train,)
        y_same: np.ndarray,             # (E_train,)
    ) -> "GraphSAGERefiner":
        """Train GraphSAGE on graph edges."""
        if len(edge_indices) == 0:
            return self

        n_nodes = len(node_features)
        t_nodes = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        t_edges = torch.tensor(edge_indices, dtype=torch.long, device=self.device)
        t_efeat = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
        t_ynext = torch.tensor(y_next, dtype=torch.float32, device=self.device)
        t_ysame = torch.tensor(y_same, dtype=torch.float32, device=self.device)
        adj_norm = self._normalize_adj(edge_indices, n_nodes)

        # Class weights
        n_pos_next = max(1, int(np.sum(y_next == 1)))
        pos_weight_next = torch.tensor([(len(y_next) - n_pos_next) / n_pos_next], device=self.device)
        pos_weight_same = torch.tensor([1.5], device=self.device)

        crit_next = nn.BCEWithLogitsLoss(pos_weight=pos_weight_next)
        crit_same = nn.BCEWithLogitsLoss(pos_weight=pos_weight_same)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # Forward
            h0 = self.model.node_proj(t_nodes)
            h1 = self.model.sage1(h0, adj_norm)
            h2 = self.model.sage2(h1, adj_norm)

            h_u = h2[t_edges[:, 0]]
            h_v = h2[t_edges[:, 1]]
            edge_cat = torch.cat([h_u, h_v, t_efeat], dim=-1)
            logits = self.model.edge_mlp(edge_cat)

            loss = crit_next(logits[:, 0], t_ynext) + crit_same(logits[:, 1], t_ysame)
            loss.backward()
            optimizer.step()

        self.model.eval()
        return self

    def predict_refined(
        self,
        node_features: np.ndarray,      # (N, node_dim)
        edge_indices: np.ndarray,       # (E, 2)
        edge_features: np.ndarray,      # (E, edge_dim)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute refined P_next and P_same for candidate edges."""
        if len(edge_indices) == 0:
            return np.zeros(0), np.zeros(0)

        self.model.eval()
        n_nodes = len(node_features)
        t_nodes = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        t_edges = torch.tensor(edge_indices, dtype=torch.long, device=self.device)
        t_efeat = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
        adj_norm = self._normalize_adj(edge_indices, n_nodes)

        with torch.no_grad():
            p_next, p_same = self.model(t_nodes, adj_norm, t_edges, t_efeat)

        return p_next.cpu().numpy(), p_same.cpu().numpy()
