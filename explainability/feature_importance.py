"""Explainable AI (doc §37): model-agnostic permutation importance,
tree-model native importance and SHAP explanations for fragment/reconstruction
decisions."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def permutation_importance_manual(model, X: np.ndarray, y: np.ndarray,
                                  feature_names: List[str],
                                  n_repeats: int = 3,
                                  score_fn=None,
                                  max_features: int = 40,
                                  sample: int = 4000,
                                  seed: int = 42) -> List[Tuple[str, float]]:
    """Model-agnostic importance via shuffle degradation of ROC-AUC (doc §37)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    if len(X) > sample:
        idx = rng.choice(len(X), sample, replace=False)
        X, y = X[idx], y[idx]
    base = roc_auc_score(y, model.predict_proba(X))

    def col_shuffle(col: np.ndarray) -> np.ndarray:
        return rng.permutation(col)

    results: List[Tuple[str, float]] = []
    n_feat = min(X.shape[1], max_features * 100)  # evaluated per family below
    for j in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, j] = col_shuffle(Xp[:, j])
        try:
            s = roc_auc_score(y, model.predict_proba(Xp))
        except Exception:
            s = base
        results.append((feature_names[j], base - s))
    results.sort(key=lambda t: -t[1])
    return results


def grouped_importance(model, X: np.ndarray, y: np.ndarray,
                       family_of: List[str]) -> Dict[str, float]:
    """Average importance per feature family (byte/statistical/structural/...)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(0)
    if len(X) > 4000:
        idx = rng.choice(len(X), 4000, replace=False)
        X, y = X[idx], y[idx]
    base = roc_auc_score(y, model.predict_proba(X))
    families = sorted(set(family_of))
    out: Dict[str, float] = {}
    for fam in families:
        cols = np.array([j for j, f in enumerate(family_of) if f == fam])
        if len(cols) == 0:
            continue
        Xp = X.copy()
        Xp[:, cols] = rng.permutation(Xp[:, cols])
        try:
            s = roc_auc_score(y, model.predict_proba(Xp))
        except Exception:
            s = base
        out[fam] = float(base - s)
    return out


def tree_feature_importance(model, feature_names: List[str],
                            top_k: int = 25) -> List[Tuple[str, float]]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None and hasattr(model, "model"):
        imp = getattr(model.model, "feature_importances_", None)
    if imp is None:
        return []
    pairs = sorted(zip(feature_names, imp.tolist()), key=lambda t: -t[1])
    return pairs[:top_k]


def shap_values(model, X: np.ndarray, feature_names: Optional[List[str]] = None,
                max_samples: int = 300) -> Tuple[np.ndarray, List[str]]:
    """SHAP values via TreeExplainer (XGBoost/RF) or KernelExplainer fallback.

    Returns (shap_matrix, feature_names); empty matrix when SHAP unavailable.
    """
    try:
        import shap
    except ImportError:
        return np.zeros((0, 0)), feature_names or []
    rng = np.random.RandomState(42)
    if len(X) > max_samples:
        Xs = X[rng.choice(len(X), max_samples, replace=False)]
    else:
        Xs = X
    inner = getattr(model, "model", model)
    try:
        explainer = shap.TreeExplainer(inner)
        vals = explainer.shap_values(Xs)
        if isinstance(vals, list):          # binary classifiers may return [class0, class1]
            vals = vals[-1]
        return np.asarray(vals), feature_names or []
    except Exception:
        pass
    try:
        background = shap.sample(X, min(100, len(X)), random_state=42)
        explainer = shap.KernelExplainer(lambda z: model.predict_proba(z), background)
        vals = explainer.shap_values(Xs[:50], nsamples="auto", silent=True)
        if isinstance(vals, list):
            vals = vals[-1]
        return np.asarray(vals), feature_names or []
    except Exception:
        return np.zeros((0, 0)), feature_names or []


def top_shap_drivers(shap_matrix: np.ndarray, feature_names: List[str],
                     sample_row: int = 0, top_k: int = 8) -> List[Dict[str, object]]:
    """Per-decision explanation: features that pushed this prediction."""
    if shap_matrix.size == 0 or shap_matrix.ndim != 2:
        return []
    row = shap_matrix[sample_row]
    order = np.argsort(-np.abs(row))[:top_k]
    return [{"feature": feature_names[j], "shap": float(row[j]),
             "direction": "supports" if row[j] > 0 else "opposes"}
            for j in order]
