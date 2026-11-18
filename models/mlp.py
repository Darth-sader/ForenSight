"""Deep-learning MLP pair classifier (doc §13): transforms pair features into
a probability, trained with PyTorch (CPU-friendly, small footprint)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import torch
import torch.nn as nn

from config import TORCH_THREADS

torch.set_num_threads(TORCH_THREADS)


class PairMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        dims = [input_dim] + [hidden] * layers + [1]
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks += [nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPTorchModel:
    name = "mlp_torch"

    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, batch_size: int = 512, epochs: int = 60,
                 patience: int = 6, seed: int = 42):
        self.cfg = dict(hidden=hidden, layers=layers, dropout=dropout, lr=lr,
                        batch_size=batch_size, epochs=epochs, patience=patience, seed=seed)
        self.module: Optional[PairMLP] = None
        self.input_dim_ = 0
        self.mean_ = None
        self.std_ = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            return X
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.cfg["seed"])
        # standardise inputs (pair features span wildly different scales)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        X = self._standardize(X)
        if X_val is not None and len(X_val):
            X_val = self._standardize(X_val)
        self.input_dim_ = X.shape[1]
        self.module = PairMLP(self.input_dim_, self.cfg["hidden"], self.cfg["layers"],
                              self.cfg["dropout"])
        opt = torch.optim.Adam(self.module.parameters(), lr=self.cfg["lr"])
        lossf = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(np.asarray(y, dtype=np.float32))
        ds = torch.utils.data.TensorDataset(Xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=True,
                                         num_workers=0)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            Xv = torch.tensor(X_val, dtype=torch.float32)
            yv = torch.tensor(np.asarray(y_val, dtype=np.float32))
        best_loss, best_state, bad = float("inf"), None, 0
        for epoch in range(self.cfg["epochs"]):
            self.module.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = lossf(self.module(xb), yb)
                loss.backward()
                opt.step()
            if has_val:
                self.module.eval()
                with torch.no_grad():
                    vloss = lossf(self.module(Xv), yv).item()
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.module is not None
        self.module.eval()
        X = self._standardize(np.asarray(X, dtype=np.float32))
        out = np.empty(len(X), dtype=np.float64)
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32)
            for i in range(0, len(X), 4096):
                out[i:i + 4096] = torch.sigmoid(self.module(xt[i:i + 4096])).numpy()
        return out

    def save(self, path: Path) -> None:
        assert self.module is not None
        torch.save({"state_dict": self.module.state_dict(), "cfg": self.cfg,
                    "input_dim": self.input_dim_, "mean": self.mean_, "std": self.std_}, path)

    def load(self, path: Path) -> "MLPTorchModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = ckpt["cfg"]
        self.input_dim_ = ckpt["input_dim"]
        self.mean_, self.std_ = ckpt.get("mean"), ckpt.get("std")
        self.module = PairMLP(self.input_dim_, self.cfg["hidden"], self.cfg["layers"],
                              self.cfg["dropout"])
        self.module.load_state_dict(ckpt["state_dict"])
        self.module.eval()
        return self
