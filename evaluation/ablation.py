"""Feature ablation studies (doc §28): cumulative feature-family experiments
A-E plus handcrafted vs learned-embedding vs hybrid representation comparison."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from common import logger, save_json
from config import RESULTS_DIR
from features import ABLATION_MASKS, PAIR_NAMES


def family_mask(families: List[str]) -> np.ndarray:
    return np.array([fam in families for _, fam in PAIR_NAMES])


def run_ablation(train_ds, val_ds, model_fn: Callable[[], object],
                 task: str = "next") -> Dict[str, object]:
    """Cumulative feature-family ablation (doc §28 experiments A-E).

    ``task`` selects the label column: "next" (task 2) or "same" (task 1).
    """
    from evaluation.metrics import classification_metrics

    y_tr = train_ds.y_next if task == "next" else train_ds.y_same
    y_va = val_ds.y_next if task == "next" else val_ds.y_same
    results: Dict[str, object] = {}
    for name, fams in ABLATION_MASKS.items():
        mask = family_mask(fams)
        model = model_fn()
        model.fit(train_ds.X[:, mask], y_tr, val_ds.X[:, mask], y_va)
        m = classification_metrics(y_va, model.predict_proba(val_ds.X[:, mask]))
        results[name] = {"families": fams, "n_features": int(mask.sum()), **m}
        logger.info("ablation %-22s dims=%-4d auc=%.4f", name, mask.sum(), m["roc_auc"])
    return results


def run_representation_comparison(train_ds, val_ds, embeddings: np.ndarray,
                                  pair_to_emb_sim: Callable[[np.ndarray], np.ndarray],
                                  model_fn: Callable[[], object],
                                  task: str = "next") -> Dict[str, object]:
    """Handcrafted vs learned embeddings vs hybrid (doc §28 bottom)."""
    from evaluation.metrics import classification_metrics

    y_tr = train_ds.y_next if task == "next" else train_ds.y_same
    y_va = val_ds.y_next if task == "next" else val_ds.y_same
    out: Dict[str, object] = {}

    # 1) handcrafted features only
    model = model_fn()
    model.fit(train_ds.X, y_tr, val_ds.X, y_va)
    out["handcrafted"] = classification_metrics(y_va, model.predict_proba(val_ds.X))

    # 2) learned representation only: embedding similarity scalar per pair
    sim_tr = pair_to_emb_sim(train_ds.pair_idx).reshape(-1, 1)
    sim_va = pair_to_emb_sim(val_ds.pair_idx).reshape(-1, 1)
    model = model_fn()
    model.fit(sim_tr.astype(np.float32), y_tr, sim_va.astype(np.float32), y_va)
    out["learned_embedding"] = classification_metrics(
        y_va, model.predict_proba(sim_va.astype(np.float32)))

    # 3) hybrid: handcrafted + embedding similarity
    Xh_tr = np.hstack([train_ds.X, sim_tr.astype(np.float32)])
    Xh_va = np.hstack([val_ds.X, sim_va.astype(np.float32)])
    model = model_fn()
    model.fit(Xh_tr, y_tr, Xh_va, y_va)
    out["hybrid"] = classification_metrics(y_va, model.predict_proba(Xh_va))
    for k, v in out.items():
        logger.info("representation %-17s auc=%.4f", k, v["roc_auc"])
    return out
