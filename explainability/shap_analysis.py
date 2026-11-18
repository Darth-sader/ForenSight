"""SHAP analysis helpers (doc §37) - thin, cached wrapper around
:func:`explainability.feature_importance.shap_values` plus summary
aggregation for plots and reports."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from explainability.feature_importance import shap_values, top_shap_drivers


def explain_pair_decision(model, X_pair: np.ndarray, feature_names: List[str],
                          row: int = 0) -> Dict[str, object]:
    """Explain one pairwise transition decision with SHAP (doc §37)."""
    vals, names = shap_values(model, X_pair, feature_names)
    return {
        "available": vals.size > 0,
        "mean_abs_shap": (float(np.abs(vals).mean(axis=0).mean()) if vals.size else None),
        "drivers": top_shap_drivers(vals, names, sample_row=row),
    }


def global_shap_summary(model, X: np.ndarray, feature_names: List[str],
                        top_k: int = 15) -> List[Dict[str, object]]:
    """Aggregate mean |SHAP| per feature = global explanation."""
    vals, names = shap_values(model, X, feature_names)
    if vals.size == 0 or vals.ndim != 2:
        return []
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_k]
    return [{"feature": names[j], "mean_abs_shap": float(mean_abs[j])} for j in order]
