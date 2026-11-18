"""Cross-validation (doc §26): stratified K-fold for classification and
group-based splitting by original file to prevent leakage."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import logger


def stratified_kfold(X: np.ndarray, y: np.ndarray, model_fn: Callable[[], object],
                     k: int = 5, seed: int = 42) -> Dict[str, object]:
    """Stratified K-fold CV for the pairwise classifiers (doc §26)."""
    from evaluation.metrics import classification_metrics

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    per_fold: List[Dict[str, float]] = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        model = model_fn()
        model.fit(X[tr], y[tr])
        m = classification_metrics(y[te], model.predict_proba(X[te]))
        per_fold.append(m)
        logger.info("fold %d: auc=%.4f f1=%.4f", fold, m["roc_auc"], m["f1"])
    agg: Dict[str, float] = {}
    for key in per_fold[0]:
        if key == "confusion":
            continue
        vals = [m[key] for m in per_fold]
        agg[key] = float(np.nanmean(vals))
        agg[key + "_std"] = float(np.nanstd(vals))
    return {"per_fold": per_fold, "aggregate": agg, "k": k}


def group_kfold_by_file(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        model_fn: Callable[[], object], k: int = 4) -> Dict[str, object]:
    """GroupKFold by source file - the same file never lands in train+test
    (doc §26 leakage prevention)."""
    from evaluation.metrics import classification_metrics

    gkf = GroupKFold(n_splits=min(k, len(np.unique(groups))))
    per_fold = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = model_fn()
        model.fit(X[tr], y[tr])
        m = classification_metrics(y[te], model.predict_proba(X[te]))
        m["n_groups_train"] = int(len(np.unique(groups[tr])))
        m["n_groups_test"] = int(len(np.unique(groups[te])))
        per_fold.append(m)
        logger.info("group-fold %d: auc=%.4f f1=%.4f", fold, m["roc_auc"], m["f1"])
    agg = {}
    for key in per_fold[0]:
        if key in ("confusion",) or key.startswith("n_groups"):
            continue
        vals = [m.get(key, np.nan) for m in per_fold]
        agg[key] = float(np.nanmean(vals))
        agg[key + "_std"] = float(np.nanstd(vals))
    return {"per_fold": per_fold, "aggregate": agg, "k": len(per_fold)}


def no_leakage_check(groups_train: np.ndarray, groups_test: np.ndarray) -> bool:
    return len(set(groups_train.tolist()) & set(groups_test.tolist())) == 0
