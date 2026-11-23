"""Graph Attention Network (GAT) Relational Refinement Model (v2.0.0-phase4).

Refines pairwise candidate transition probabilities using multi-head graph attention:
1. Fragment node projections Wh_u in R^{d_hidden}.
2. Attention Mechanism:
       e_{uv} = LeakyReLU(a^T [Wh_u || Wh_v || W_e e_feat])
       alpha_{uv} = Softmax_{v in N(u)}(e_{uv})
3. Attentional Neighborhood Aggregation:
       h_u' = ELU( sum_{v in N(u)} alpha_{uv} W h_v )
4. Multi-Task Edge Scoring Head:
       [h_u' || h_v' || e_feat] -> (P_next_refined, P_same_refined)
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


class GATLayer(nn.Module):
    """Multi-Head Graph Attention Layer with edge feature conditioning."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 6,
        n_heads: int = 2,
        dropout: float = 0.15,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.w_node = nn.Linear(in_dim, out_dim, bias=False)
        self.w_edge = nn.Linear(edge_dim, out_dim, bias=False)
        self.attn_weights = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim + self.head_dim))
        nn.init.xavier_uniform_(self.attn_weights)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,             # (N, in_dim)
        edge_indices: torch.Tensor,  # (E, 2)
        edge_features: torch.Tensor, # (E, edge_dim)
    ) -> torch.Tensor:
        n = x.size(0)
        h = self.w_node(x).view(n, self.n_heads, self.head_dim)          # (N, H, D)
        e_proj = self.w_edge(edge_features).view(-1, self.n_heads, self.head_dim) # (E, H, D)

        u = edge_indices[:, 0]
        v = edge_indices[:, 1]

        # Concatenate [h_u || h_v || e_proj] per edge
        cat_edges = torch.cat([h[u], h[v], e_proj], dim=-1)  # (E, H, 3D)

        # Compute unnormalized attention scores
        # (E, H, 3D) * (H, 3D) -> sum -> (E, H)
        attn_logits = self.leaky_relu((cat_edges * self.attn_weights.unsqueeze(0)).sum(dim=-1))

        # Numerically stable edge softmax grouped by source node u
        attn_exp = torch.exp(attn_logits - attn_logits.max())
        denom = torch.zeros(n, self.n_heads, device=x.device).scatter_add_(
            0, u.unsqueeze(-1).expand(-1, self.n_heads), attn_exp
        )
        attn_weights = attn_exp / (denom[u] + 1e-12)
        attn_weights = self.dropout(attn_weights)  # (E, H)

        # Message aggregation: weighted sum of h[v]
        msg = (h[v] * attn_weights.unsqueeze(-1)).view(-1, self.out_dim)
        out = torch.zeros(n, self.out_dim, device=x.device).scatter_add_(
            0, u.unsqueeze(-1).expand(-1, self.out_dim), msg
        )

        return F.elu(out) + self.w_node(x)  # Residual connection


class GATNet(nn.Module):
    """Complete 2-layer Graph Attention Network with edge prediction head."""

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        edge_dim: int = 6,
        n_heads: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gat1 = GATLayer(hidden_dim, hidden_dim, edge_dim=edge_dim, n_heads=n_heads, dropout=dropout)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, edge_dim=edge_dim, n_heads=n_heads, dropout=dropout)

        edge_in = (hidden_dim * 2) + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = self.proj(node_features)
        h1 = self.gat1(h0, edge_indices, edge_features)
        h2 = self.gat2(h1, edge_indices, edge_features)

        u = edge_indices[:, 0]
        v = edge_indices[:, 1]
        cat_e = torch.cat([h2[u], h2[v], edge_features], dim=-1)
        logits = self.edge_mlp(cat_e)

        p_next = torch.sigmoid(logits[:, 0])
        p_same = torch.sigmoid(logits[:, 1])
        return p_next, p_same


class GATRefiner:
    """Trainer and inference interface for GAT relational edge refinement."""

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        edge_dim: int = 6,
        n_heads: int = 2,
        lr: float = 2e-3,
        epochs: int = 40,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = GATNet(node_dim=node_dim, hidden_dim=hidden_dim, edge_dim=edge_dim, n_heads=n_heads).to(self.device)
        self.lr = lr
        self.epochs = epochs

    def fit(
        self,
        node_features: np.ndarray,
        edge_indices: np.ndarray,
        edge_features: np.ndarray,
        y_next: np.ndarray,
        y_same: np.ndarray,
    ) -> "GATRefiner":
        if len(edge_indices) == 0:
            return self

        t_nodes = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        t_edges = torch.tensor(edge_indices, dtype=torch.long, device=self.device)
        t_efeat = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
        t_ynext = torch.tensor(y_next, dtype=torch.float32, device=self.device)
        t_ysame = torch.tensor(y_same, dtype=torch.float32, device=self.device)

        n_pos_next = max(1, int(np.sum(y_next == 1)))
        pos_weight_next = torch.tensor([(len(y_next) - n_pos_next) / n_pos_next], device=self.device)
        crit_next = nn.BCEWithLogitsLoss(pos_weight=pos_weight_next)
        crit_same = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            h0 = self.model.proj(t_nodes)
            h1 = self.model.gat1(h0, t_edges, t_efeat)
            h2 = self.model.gat2(h1, t_edges, t_efeat)

            u = t_edges[:, 0]
            v = t_edges[:, 1]
            cat_e = torch.cat([h2[u], h2[v], t_efeat], dim=-1)
            logits = self.model.edge_mlp(cat_e)

            loss = crit_next(logits[:, 0], t_ynext) + crit_same(logits[:, 1], t_ysame)
            loss.backward()
            optimizer.step()

        self.model.eval()
        return self

    def predict_refined(
        self,
        node_features: np.ndarray,
        edge_indices: np.ndarray,
        edge_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(edge_indices) == 0:
            return np.zeros(0), np.zeros(0)

        self.model.eval()
        t_nodes = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        t_edges = torch.tensor(edge_indices, dtype=torch.long, device=self.device)
        t_efeat = torch.tensor(edge_features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            p_next, p_same = self.model(t_nodes, t_edges, t_efeat)

        return p_next.cpu().numpy(), p_same.cpu().numpy()
