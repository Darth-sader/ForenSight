"""Hyperparameter optimization (doc §25): grid, random and Bayesian search
over the supervised models using validation data only (never the test set).

The search space is deliberately compact so the whole pipeline stays light on
system resources while still exercising all three search strategies.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from common import logger

SEARCH_SPACES: Dict[str, List[dict]] = {
    "random_forest": [
        {"n_estimators": v, "max_depth": d, "min_samples_split": s}
        for v, d, s in itertools.product([120, 200, 320], [16, 24, None], [2, 4, 8])
    ],
    "xgboost": [
        {"learning_rate": lr, "n_estimators": n, "max_depth": d, "subsample": ss,
         "colsample_bytree": cs}
        for lr, n, d, ss, cs in itertools.product(
            [0.05, 0.1, 0.2], [200, 300, 450], [4, 6, 8], [0.8, 0.95], [0.7, 0.9])
    ],
    "mlp_torch": [
        {"hidden": h, "layers": l, "dropout": dr, "lr": lr}
        for h, l, dr, lr in itertools.product([96, 128, 192], [1, 2, 3],
                                              [0.1, 0.2], [5e-4, 1e-3])
    ],
    "logistic_regression": [
        {"C": c} for c in [0.1, 0.5, 1.0, 2.0, 5.0]
    ],
}


def _make_model(family: str, params: dict, seed: int):
    if family == "random_forest":
        from models.random_forest import RandomForestModel
        return RandomForestModel(seed=seed, **params)
    if family == "xgboost":
        from models.xgboost import XGBoostModel
        return XGBoostModel(seed=seed, **params)
    if family == "mlp_torch":
        from models.mlp import MLPTorchModel
        return MLPTorchModel(seed=seed, **params)
    if family == "logistic_regression":
        from models.logistic import LogisticModel
        return LogisticModel(seed=seed, **params)
    raise ValueError(family)


def random_search(family: str, X_tr, y_tr, X_val, y_val, n_trials: int = 8,
                  seed: int = 42) -> Tuple[dict, float, List[dict]]:
    rng = random.Random(seed)
    space = SEARCH_SPACES[family]
    trials = rng.sample(space, min(n_trials, len(space)))
    best_params, best_auc, log = None, -1.0, []
    for params in trials:
        model = _make_model(family, params, seed)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = roc_auc_score(y_val, model.predict_proba(X_val))
        log.append({"params": params, "val_auc": float(auc)})
        if auc > best_auc:
            best_params, best_auc = params, float(auc)
    logger.info("random_search %s -> auc=%.4f params=%s", family, best_auc, best_params)
    return best_params, best_auc, log


def grid_search(family: str, X_tr, y_tr, X_val, y_val,
                max_configs: int = 12, seed: int = 42) -> Tuple[dict, float, List[dict]]:
    space = SEARCH_SPACES[family][:max_configs]
    best_params, best_auc, log = None, -1.0, []
    for params in space:
        model = _make_model(family, params, seed)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = roc_auc_score(y_val, model.predict_proba(X_val))
        log.append({"params": params, "val_auc": float(auc)})
        if auc > best_auc:
            best_params, best_auc = params, float(auc)
    logger.info("grid_search %s -> auc=%.4f params=%s", family, best_auc, best_params)
    return best_params, best_auc, log


def _gp_posterior_mean(X_hist: np.ndarray, y_hist: np.ndarray, X_cand: np.ndarray,
                       length_scale: float = 0.3) -> np.ndarray:
    """Tiny GP surrogate (RBF kernel) for Bayesian optimisation expected score."""
    def rbf(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * length_scale ** 2))
    K = rbf(X_hist, X_hist) + 1e-6 * np.eye(len(X_hist))
    Ks = rbf(X_hist, X_cand)
    alpha = np.linalg.solve(K, y_hist)
    return Ks.T @ alpha


def param_vector(family: str, p: dict) -> List[float]:
    keys = {
        "random_forest": ["n_estimators", "max_depth", "min_samples_split"],
        "xgboost": ["learning_rate", "n_estimators", "max_depth", "subsample", "colsample_bytree"],
        "mlp_torch": ["hidden", "layers", "dropout", "lr"],
        "logistic_regression": ["C"],
    }[family]
    return [float(p[k] if p[k] is not None else 64) for k in keys]


def bayesian_search(family: str, X_tr, y_tr, X_val, y_val, n_trials: int = 8,
                    n_initial: int = 3, seed: int = 42) -> Tuple[dict, float, List[dict]]:
    """Bayesian optimisation with a GP surrogate over the categorical space."""
    rng = random.Random(seed)
    space = SEARCH_SPACES[family]
    rng.shuffle(space)
    log: List[dict] = []
    X_hist: List[List[float]] = []
    y_hist: List[float] = []
    best_params, best_auc = None, -1.0
    for params in space[:n_initial]:          # initial random design
        model = _make_model(family, params, seed)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = roc_auc_score(y_val, model.predict_proba(X_val))
        log.append({"params": params, "val_auc": float(auc), "strategy": "initial"})
        X_hist.append(param_vector(family, params))
        y_hist.append(auc)
        if auc > best_auc:
            best_params, best_auc = params, float(auc)
    remaining = [p for p in space if p not in space[:n_initial]]
    while len(log) < n_trials and remaining:
        Xc = np.array([param_vector(family, p) for p in remaining])
        mu = _gp_posterior_mean(np.array(X_hist), np.array(y_hist), Xc)
        pick = int(np.argmax(mu))
        params = remaining.pop(pick)
        model = _make_model(family, params, seed)
        model.fit(X_tr, y_tr, X_val, y_val)
        auc = roc_auc_score(y_val, model.predict_proba(X_val))
        log.append({"params": params, "val_auc": float(auc), "strategy": "gp-acquisition"})
        X_hist.append(param_vector(family, params))
        y_hist.append(auc)
        if auc > best_auc:
            best_params, best_auc = params, float(auc)
    logger.info("bayesian_search %s -> auc=%.4f params=%s", family, best_auc, best_params)
    return best_params, best_auc, log


def tune_model(family: str, X_tr, y_tr, X_val, y_val, strategy: str = "bayesian",
               n_trials: int = 8, seed: int = 42):
    """Entry point: tune one model family with the requested strategy."""
    if strategy == "grid":
        return grid_search(family, X_tr, y_tr, X_val, y_val, max_configs=n_trials, seed=seed)
    if strategy == "random":
        return random_search(family, X_tr, y_tr, X_val, y_val, n_trials=n_trials, seed=seed)
    return bayesian_search(family, X_tr, y_tr, X_val, y_val, n_trials=n_trials, seed=seed)
