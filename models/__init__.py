"""Model package (doc §12-14) with an ensemble wrapper (doc §5 ensemble learning)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from models.logistic import LogisticModel
from models.mlp import MLPTorchModel
from models.random_forest import RandomForestModel
from models.siamese import SiameseNetwork
from models.xgboost import XGBoostModel

__all__ = ["LogisticModel", "RandomForestModel", "XGBoostModel", "MLPTorchModel",
           "SiameseNetwork", "TransitionEnsemble", "build_tabular_models"]


def build_tabular_models(seed: int = 42) -> List:
    return [LogisticModel(seed=seed), RandomForestModel(seed=seed),
            XGBoostModel(seed=seed), MLPTorchModel(seed=seed)]


class TransitionEnsemble:
    """Weighted probability ensemble over the fitted supervised models.

    Weights derive from validation ROC-AUC (empirically calibrated rather than
    arbitrary, doc §20) and the combined probability is used as the ML
    transition score in reconstruction scoring.
    """

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.weights: Dict[str, float] = {}
        self.val_auc: Dict[str, float] = {}

    def add(self, name: str, model, val_auc: float) -> None:
        self.models[name] = model
        self.val_auc[name] = float(val_auc)

    def finalize(self) -> "TransitionEnsemble":
        aucs = {k: max(0.5, v) for k, v in self.val_auc.items()}
        total = sum((aucs[k] - 0.5) for k in aucs) or 1.0
        self.weights = {k: (aucs[k] - 0.5) / total for k in aucs}
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X))
        for name, model in self.models.items():
            out += self.weights[name] * model.predict_proba(X)
        return np.clip(out, 0.0, 1.0)

    def predict_proba_per_model(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {name: model.predict_proba(X) for name, model in self.models.items()}

    def summary(self) -> Dict[str, object]:
        return {"models": dict(self.val_auc), "weights": {k: round(v, 4) for k, v in self.weights.items()}}
