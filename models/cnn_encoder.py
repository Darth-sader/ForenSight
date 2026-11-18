"""Raw-byte 1D CNN representation learning.

A convolutional Siamese encoder that reads the *raw fragment bytes* (not
handcrafted features) and learns a compact embedding with contrastive +
triplet losses.  This gives ForenSight a second, independent representation
source alongside the handcrafted-feature MLP encoder; the two embedding
spaces are fused (average cosine similarity) for candidate generation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBEDDING_DIM, SIAMESE_MARGIN, TORCH_THREADS

torch.set_num_threads(TORCH_THREADS)


class RawByteCNN(nn.Module):
    """1D CNN over normalised raw bytes: (B, 1, block_size) -> (B, emb_dim)."""

    def __init__(self, block_size: int = 512, emb_dim: int = EMBEDDING_DIM,
                 channels: tuple = (32, 64, 96), fc_hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=8, stride=4, padding=2), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=6, stride=3, padding=1), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(c3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(c3 * 4, fc_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_hidden, emb_dim),
        )

    def forward(self, raw_bytes: torch.Tensor) -> torch.Tensor:
        # raw_bytes: (B, block) uint8-ish floats
        x = (raw_bytes / 255.0).unsqueeze(1)            # (B, 1, block)
        z = self.conv(x).flatten(1)
        z = self.fc(z)
        return F.normalize(z, p=2, dim=-1)


class SiameseCNN:
    """Siamese wrapper training :class:`RawByteCNN` on raw fragment bytes."""

    name = "siamese_cnn"

    def __init__(self, block_size: int = 512, emb_dim: int = EMBEDDING_DIM,
                 margin: float = SIAMESE_MARGIN, lr: float = 1e-3,
                 batch_size: int = 256, epochs: int = 25, patience: int = 4,
                 triplet_weight: float = 0.5, seed: int = 42):
        self.cfg = dict(block_size=block_size, emb_dim=emb_dim, margin=margin, lr=lr,
                        batch_size=batch_size, epochs=epochs, patience=patience,
                        triplet_weight=triplet_weight, seed=seed)
        self.module = RawByteCNN(block_size, emb_dim)

    # ------------------------------------------------------------------ fit
    def fit(self, raw_matrix: np.ndarray, pairs: np.ndarray, labels_same: np.ndarray,
            labels_next: np.ndarray, val_raw: Optional[np.ndarray] = None,
            val_pairs: Optional[np.ndarray] = None, val_labels_same: Optional[np.ndarray] = None):
        torch.manual_seed(self.cfg["seed"])
        opt = torch.optim.Adam(self.module.parameters(), lr=self.cfg["lr"])
        X = torch.tensor(raw_matrix, dtype=torch.float32)
        pairs_t = torch.tensor(np.asarray(pairs), dtype=torch.long)
        same_t = torch.tensor(np.asarray(labels_same, dtype=np.float32))
        next_t = torch.tensor(np.asarray(labels_next, dtype=np.float32))
        n = len(X)

        has_val = val_raw is not None and val_pairs is not None
        if has_val:
            Xv = torch.tensor(val_raw, dtype=torch.float32)
            pv = torch.tensor(np.asarray(val_pairs), dtype=torch.long)
            sv = torch.tensor(np.asarray(val_labels_same, dtype=np.float32))

        best_loss, best_state, bad = float("inf"), None, 0
        for epoch in range(self.cfg["epochs"]):
            self.module.train()
            perm = torch.randperm(n)
            for start in range(0, len(pairs_t), self.cfg["batch_size"]):
                pb = pairs_t[start:start + self.cfg["batch_size"]]
                if len(pb) < 8:
                    continue
                sb = same_t[start:start + self.cfg["batch_size"]]
                za = self.module(X[pb[:, 0]])
                zb = self.module(X[pb[:, 1]])
                dist = F.pairwise_distance(za, zb)
                contrastive = torch.mean(
                    sb * dist ** 2 + (1 - sb) * F.relu(self.cfg["margin"] - dist) ** 2)
                neg_idx = torch.randint(0, n, (len(pb),))
                zn = self.module(X[neg_idx])
                triplet = F.relu(F.pairwise_distance(za, zb) -
                                 F.pairwise_distance(za, zn) + self.cfg["margin"]).mean()
                loss = contrastive + self.cfg["triplet_weight"] * triplet
                opt.zero_grad()
                loss.backward()
                opt.step()
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
    def embed(self, raw_matrix: np.ndarray, batch: int = 2048) -> np.ndarray:
        self.module.eval()
        X = torch.tensor(raw_matrix, dtype=torch.float32)
        out = np.empty((len(X), self.cfg["emb_dim"]), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(X), batch):
                out[i:i + batch] = self.module(X[i:i + batch]).numpy()
        return out

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.module.state_dict(), "cfg": self.cfg}, path)

    def load(self, path: Path) -> "SiameseCNN":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = ckpt["cfg"]
        self.module = RawByteCNN(self.cfg["block_size"], self.cfg["emb_dim"])
        self.module.load_state_dict(ckpt["state_dict"])
        self.module.eval()
        return self
