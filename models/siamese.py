"""Siamese / metric-learning encoder (doc §14): learns a fragment embedding
space where related fragments are close and unrelated fragments are far, using
contrastive loss (same-file pairs) + triplet loss (consecutive triples).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDING_DIM, SIAMESE_MARGIN, TORCH_THREADS

torch.set_num_threads(TORCH_THREADS)


class FragmentEncoder(nn.Module):
    """Neural encoder: single-fragment features -> compact embedding (doc §13)."""

    def __init__(self, input_dim: int, hidden: int = 192, emb_dim: int = EMBEDDING_DIM,
                 layers: int = 2, dropout: float = 0.15):
        super().__init__()
        dims = [input_dim] + [hidden] * layers
        blocks = []
        for i in range(len(dims) - 1):
            blocks += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                       nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.head(z)
        return F.normalize(z, p=2, dim=-1)


class SiameseNetwork:
    """Wraps :class:`FragmentEncoder` with contrastive + triplet training."""

    name = "siamese_encoder"

    def __init__(self, input_dim: int, emb_dim: int = EMBEDDING_DIM,
                 margin: float = SIAMESE_MARGIN, lr: float = 1e-3,
                 batch_size: int = 512, epochs: int = 40, patience: int = 5,
                 triplet_weight: float = 0.5, seed: int = 42):
        self.cfg = dict(input_dim=input_dim, emb_dim=emb_dim, margin=margin, lr=lr,
                        batch_size=batch_size, epochs=epochs, patience=patience,
                        triplet_weight=triplet_weight, seed=seed)
        self.module = FragmentEncoder(input_dim, emb_dim=emb_dim)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            return X
        return ((X - self.mean_) / self.std_).astype(np.float32)

    # ------------------------------------------------------------------ fit
    def fit(self, single_matrix: np.ndarray, pairs: np.ndarray, labels_same: np.ndarray,
            labels_next: np.ndarray, X_val_single: Optional[np.ndarray] = None,
            val_pairs: Optional[np.ndarray] = None, val_labels_same: Optional[np.ndarray] = None):
        torch.manual_seed(self.cfg["seed"])
        # standardise handcrafted features (LayerNorm alone is not enough)
        self.mean_ = single_matrix.mean(axis=0)
        self.std_ = single_matrix.std(axis=0) + 1e-6
        opt = torch.optim.Adam(self.module.parameters(), lr=self.cfg["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg["epochs"])
        X = torch.tensor(self._standardize(single_matrix), dtype=torch.float32)
        n = len(X)
        pairs_t = torch.tensor(np.asarray(pairs), dtype=torch.long)
        same_t = torch.tensor(np.asarray(labels_same, dtype=np.float32))
        next_t = torch.tensor(np.asarray(labels_next, dtype=np.float32))

        has_val = X_val_single is not None and val_pairs is not None
        if has_val:
            Xv = torch.tensor(self._standardize(X_val_single), dtype=torch.float32)
            pv = torch.tensor(np.asarray(val_pairs), dtype=torch.long)
            sv = torch.tensor(np.asarray(val_labels_same, dtype=np.float32))

        best_loss, best_state, bad = float("inf"), None, 0
        for epoch in range(self.cfg["epochs"]):
            self.module.train()
            perm = torch.randperm(n)
            total = 0.0
            for start in range(0, len(pairs_t), self.cfg["batch_size"]):
                pb = pairs_t[start:start + self.cfg["batch_size"]]
                sb = same_t[start:start + self.cfg["batch_size"]]
                nb = next_t[start:start + self.cfg["batch_size"]]
                if len(pb) < 8:
                    continue
                # contrastive on sampled pairs
                za = self.module(X[pb[:, 0]])
                zb = self.module(X[pb[:, 1]])
                dist = F.pairwise_distance(za, zb)
                contrastive = torch.mean(
                    sb * dist ** 2 + (1 - sb) * F.relu(self.cfg["margin"] - dist) ** 2)
                # triplet on consecutive pairs: anchor, positive successor,
                # negative = embedding of a random other fragment
                neg_idx = torch.randint(0, n, (len(pb),))
                zn = self.module(X[neg_idx])
                pos_dist = F.pairwise_distance(za, zb)
                neg_dist = F.pairwise_distance(za, zn)
                triplet = F.relu(pos_dist - neg_dist + self.cfg["margin"]).mean()
                loss = contrastive + self.cfg["triplet_weight"] * triplet
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(pb)
            sched.step()
            if has_val:
                self.module.eval()
                with torch.no_grad():
                    za, zb = self.module(Xv[pv[:, 0]]), self.module(Xv[pv[:, 1]])
                    d = F.pairwise_distance(za, zb)
                    vloss = torch.mean(sv * d ** 2 +
                                       (1 - sv) * F.relu(self.cfg["margin"] - d) ** 2).item()
                if vloss < best_loss - 1e-4:
                    best_loss, bad = vloss, 0
                    best_state = {k: v.clone() for k, v in self.module.state_dict().items()}
                else:
                    bad += 1
                if bad >= self.cfg["patience"]:
                    break
        if best_state is not None:
            self.module.load_state_dict(best_state)
        self.module.eval()
        return self

    # -------------------------------------------------------------- embed
    def embed(self, single_matrix: np.ndarray, batch: int = 4096) -> np.ndarray:
        self.module.eval()
        X = torch.tensor(self._standardize(single_matrix), dtype=torch.float32)
        out = np.empty((len(X), self.cfg["emb_dim"]), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(X), batch):
                out[i:i + batch] = self.module(X[i:i + batch]).numpy()
        return out

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.module.state_dict(), "cfg": self.cfg,
                    "mean": self.mean_, "std": self.std_}, path)

    def load(self, path: Path) -> "SiameseNetwork":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = ckpt["cfg"]
        self.mean_, self.std_ = ckpt.get("mean"), ckpt.get("std")
        self.module = FragmentEncoder(self.cfg["input_dim"], emb_dim=self.cfg["emb_dim"])
        self.module.load_state_dict(ckpt["state_dict"])
        self.module.eval()
        return self
