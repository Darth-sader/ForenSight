"""Shared-Encoder Multi-Task Dual Relation Model (v2.0.0-phase3).

Implements a unified multi-task neural network with:
1. Shared fragment / pairwise representation trunk Phi(x)
2. Symmetric same-file membership head:
       P_same(i, j) = sigmoid(g_same(Phi(x_sym)))
   guaranteeing mathematical symmetry P_same(i, j) == P_same(j, i).
3. Directed next-fragment transition head:
       P_next(i, j) = sigmoid(g_next(Phi(x_dir)))
   preserving asymmetric sequence order P_next(i, j) != P_next(j, i).
4. Joint multi-task objective:
       L = lambda_same * L_same + lambda_next * L_next
   with validation-tuned task weights.
5. Platt scaling calibration on disjoint calibration data.
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
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from common import logger


class MultiTaskNet(nn.Module):
    """PyTorch multi-task neural network with shared encoder and dual task heads."""

    def __init__(
        self,
        input_dim: int = 362,
        shared_dim: int = 128,
        head_hidden: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Shared feature encoder / trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, shared_dim * 2),
            nn.BatchNorm1d(shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task 1 Head: Symmetric same-file membership
        self.same_head = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        # Task 2 Head: Directed asymmetric next-fragment transition
        self.next_head = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        x_sym: torch.Tensor,
        x_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute task logits using shared trunk.

        Args:
            x_sym: Symmetric feature representation (i, j symmetric)
            x_dir: Directed feature representation (i -> j directed)

        Returns:
            (same_logits, next_logits)
        """
        z_sym = self.shared_trunk(x_sym)
        z_dir = self.shared_trunk(x_dir)

        same_logits = self.same_head(z_sym).squeeze(-1)
        next_logits = self.next_head(z_dir).squeeze(-1)

        return same_logits, next_logits


class MultiTaskDualModel:
    """Wrapper providing fit, calibration, and inference for the multi-task dual model."""

    name = "multitask_dual_model"

    def __init__(
        self,
        input_dim: int = 362,
        shared_dim: int = 128,
        head_hidden: int = 64,
        lambda_same: float = 1.0,
        lambda_next: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 30,
        dropout: float = 0.15,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.head_hidden = head_hidden
        self.lambda_same = lambda_same
        self.lambda_next = lambda_next
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.net = MultiTaskNet(
            input_dim=input_dim,
            shared_dim=shared_dim,
            head_hidden=head_hidden,
            dropout=dropout,
        ).to(self.device)

        self.same_calibrator_: Optional[LogisticRegression] = None
        self.next_calibrator_: Optional[LogisticRegression] = None

    def _make_symmetric_features(self, X_fwd: np.ndarray, X_rev: Optional[np.ndarray] = None) -> np.ndarray:
        """Construct permutation-invariant symmetric feature matrix."""
        if X_rev is None:
            return X_fwd
        return 0.5 * (X_fwd + X_rev)

    def fit(
        self,
        X_train_fwd: np.ndarray,
        y_train_same: np.ndarray,
        y_train_next: np.ndarray,
        X_train_rev: Optional[np.ndarray] = None,
        X_val_fwd: Optional[np.ndarray] = None,
        y_val_same: Optional[np.ndarray] = None,
        y_val_next: Optional[np.ndarray] = None,
        X_val_rev: Optional[np.ndarray] = None,
        X_cal_fwd: Optional[np.ndarray] = None,
        y_cal_same: Optional[np.ndarray] = None,
        y_cal_next: Optional[np.ndarray] = None,
        X_cal_rev: Optional[np.ndarray] = None,
    ) -> "MultiTaskDualModel":
        """Train multi-task neural network with joint loss and calibrate on calibration partition."""
        if X_train_fwd.shape[1] != self.input_dim:
            self.input_dim = X_train_fwd.shape[1]
            self.net = MultiTaskNet(
                input_dim=self.input_dim,
                shared_dim=self.shared_dim,
                head_hidden=self.head_hidden,
                dropout=self.dropout,
            ).to(self.device)

        X_train_sym = self._make_symmetric_features(X_train_fwd, X_train_rev)
        
        # Positive class weighting for imbalanced tasks
        n_pos_same = max(1, int(np.sum(y_train_same == 1)))
        n_neg_same = max(1, int(np.sum(y_train_same == 0)))
        pos_weight_same = torch.tensor([n_neg_same / n_pos_same], device=self.device, dtype=torch.float32)

        n_pos_next = max(1, int(np.sum(y_train_next == 1)))
        n_neg_next = max(1, int(np.sum(y_train_next == 0)))
        pos_weight_next = torch.tensor([n_neg_next / n_pos_next], device=self.device, dtype=torch.float32)

        criterion_same = nn.BCEWithLogitsLoss(pos_weight=pos_weight_same)
        criterion_next = nn.BCEWithLogitsLoss(pos_weight=pos_weight_next)

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        dataset = TensorDataset(
            torch.tensor(X_train_sym, dtype=torch.float32),
            torch.tensor(X_train_fwd, dtype=torch.float32),
            torch.tensor(y_train_same, dtype=torch.float32),
            torch.tensor(y_train_next, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_sym, batch_dir, batch_ysame, batch_ynext in loader:
                batch_sym = batch_sym.to(self.device)
                batch_dir = batch_dir.to(self.device)
                batch_ysame = batch_ysame.to(self.device)
                batch_ynext = batch_ynext.to(self.device)

                optimizer.zero_grad()
                logits_same, logits_next = self.net(batch_sym, batch_dir)

                loss_same = criterion_same(logits_same, batch_ysame)
                loss_next = criterion_next(logits_next, batch_ynext)
                loss = (self.lambda_same * loss_same) + (self.lambda_next * loss_next)

                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())

        # Fit Platt scaling calibrators on disjoint calibration split
        if X_cal_fwd is not None and y_cal_same is not None and y_cal_next is not None and len(X_cal_fwd) > 0:
            X_cal_sym = self._make_symmetric_features(X_cal_fwd, X_cal_rev)
            logits_same_cal, logits_next_cal = self._predict_logits(X_cal_sym, X_cal_fwd)

            self.same_calibrator_ = LogisticRegression(C=1.0, solver="lbfgs")
            self.same_calibrator_.fit(logits_same_cal.reshape(-1, 1), y_cal_same)

            self.next_calibrator_ = LogisticRegression(C=1.0, solver="lbfgs")
            self.next_calibrator_.fit(logits_next_cal.reshape(-1, 1), y_cal_next)

        return self

    def _predict_logits(self, X_sym: np.ndarray, X_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute raw model logits."""
        self.net.eval()
        t_sym = torch.tensor(X_sym, dtype=torch.float32, device=self.device)
        t_dir = torch.tensor(X_dir, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            l_same, l_next = self.net(t_sym, t_dir)
        return l_same.cpu().numpy(), l_next.cpu().numpy()

    def predict_raw(
        self,
        X_fwd: np.ndarray,
        X_rev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute uncalibrated probabilities for same-file, forward next, and reverse next.

        Returns:
            (p_same_raw, p_next_raw, p_prev_raw)
        """
        X_sym = self._make_symmetric_features(X_fwd, X_rev)
        l_same, l_next = self._predict_logits(X_sym, X_fwd)
        p_same_raw = 1.0 / (1.0 + np.exp(-l_same))
        p_next_raw = 1.0 / (1.0 + np.exp(-l_next))

        if X_rev is not None:
            _, l_prev = self._predict_logits(X_sym, X_rev)
            p_prev_raw = 1.0 / (1.0 + np.exp(-l_prev))
        else:
            p_prev_raw = p_next_raw

        return p_same_raw, p_next_raw, p_prev_raw

    def predict_proba(
        self,
        X_fwd: np.ndarray,
        X_rev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute calibrated probabilities for same-file, forward next, and reverse next.

        Returns:
            (p_same_cal, p_next_cal, p_prev_cal)
        """
        X_sym = self._make_symmetric_features(X_fwd, X_rev)
        l_same, l_next = self._predict_logits(X_sym, X_fwd)

        if self.same_calibrator_ is not None:
            p_same_cal = self.same_calibrator_.predict_proba(l_same.reshape(-1, 1))[:, 1]
        else:
            p_same_cal = 1.0 / (1.0 + np.exp(-l_same))

        if self.next_calibrator_ is not None:
            p_next_cal = self.next_calibrator_.predict_proba(l_next.reshape(-1, 1))[:, 1]
        else:
            p_next_cal = 1.0 / (1.0 + np.exp(-l_next))

        if X_rev is not None:
            _, l_prev = self._predict_logits(X_sym, X_rev)
            if self.next_calibrator_ is not None:
                p_prev_cal = self.next_calibrator_.predict_proba(l_prev.reshape(-1, 1))[:, 1]
            else:
                p_prev_cal = 1.0 / (1.0 + np.exp(-l_prev))
        else:
            p_prev_cal = p_next_cal

        return p_same_cal, p_next_cal, p_prev_cal

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "shared_dim": self.shared_dim,
                "head_hidden": self.head_hidden,
                "lambda_same": self.lambda_same,
                "lambda_next": self.lambda_next,
                "dropout": self.dropout,
                "seed": self.seed,
            },
            "same_calibrator": self.same_calibrator_,
            "next_calibrator": self.next_calibrator_,
        }, p)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "MultiTaskDualModel":
        data = torch.load(path, map_location=device or "cpu")
        cfg = data["config"]
        model = cls(
            input_dim=cfg["input_dim"],
            shared_dim=cfg["shared_dim"],
            head_hidden=cfg["head_hidden"],
            lambda_same=cfg["lambda_same"],
            lambda_next=cfg["lambda_next"],
            dropout=cfg.get("dropout", 0.15),
            seed=cfg.get("seed", 42),
            device=device,
        )
        model.net.load_state_dict(data["state_dict"])
        model.same_calibrator_ = data.get("same_calibrator")
        model.next_calibrator_ = data.get("next_calibrator")
        model.net.eval()
        return model
