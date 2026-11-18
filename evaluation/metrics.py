"""Evaluation metrics (doc §35): classification, reconstruction, anomaly,
calibration and ranking metrics."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray,
                           threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    y_true = y_true.astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    except ValueError:
        out["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def reconstruction_metrics(true_sequences: List[List[str]],
                           pred_sequences: List[List[str]],
                           gap_truth: Optional[Dict[str, List[int]]] = None) -> Dict[str, float]:
    """Fragment placement / transition / reconstruction rates (doc §35)."""
    n = len(true_sequences)
    if n == 0:
        return {}
    placement_scores, transition_scores, complete, partial, false_recon = [], [], 0, 0, 0
    gap_hits, gap_total = 0, 0
    matched: set = set()
    for true_seq in true_sequences:
        tset = set(true_seq)
        best_overlap, best_pred, best_transitions = 0.0, None, 0
        for pred in pred_sequences:
            key = (tuple(true_seq), tuple(pred))
            if key in matched:
                continue
            pset = set(pred)
            overlap = len(tset & pset) / max(1, len(tset))
            if overlap > best_overlap:
                best_overlap, best_pred = overlap, pred
        if best_pred is not None:
            matched.add((tuple(true_seq), tuple(best_pred)))
            placement_scores.append(best_overlap)
            # transition accuracy: fraction of consecutive pairs in truth found in pred
            true_pairs = {(true_seq[i], true_seq[i + 1]) for i in range(len(true_seq) - 1)}
            pred_pairs = {(best_pred[i], best_pred[i + 1]) for i in range(len(best_pred) - 1)}
            trans = len(true_pairs & pred_pairs) / max(1, len(true_pairs))
            transition_scores.append(trans)
            if best_overlap >= 0.999:
                complete += 1
            elif best_overlap >= 0.5:
                partial += 1
            if gap_truth and true_seq[0] in gap_truth:
                gaps = gap_truth[true_seq[0]]
                gap_total += len(gaps)
        else:
            placement_scores.append(0.0)
            transition_scores.append(0.0)
    false_recon = max(0, len(pred_sequences) - len(matched))
    return {
        "fragment_placement_accuracy": float(np.mean(placement_scores)),
        "transition_accuracy": float(np.mean(transition_scores)),
        "complete_reconstruction_rate": complete / n,
        "partial_reconstruction_rate": partial / n,
        "false_reconstruction_rate": false_recon / max(1, len(pred_sequences)),
        "n_true": n,
        "n_pred": len(pred_sequences),
    }


def topk_recovery(true_sequences: List[List[str]], pred_ranked: List[List[str]],
                  ks: Sequence[int] = (1, 3, 5)) -> Dict[str, float]:
    """Top-K recovery: fraction of true files found among the first K candidates
    with >= 80% fragment overlap (doc §35)."""
    out: Dict[str, float] = {}
    for k in ks:
        hits = 0
        for true_seq in true_sequences:
            tset = set(true_seq)
            found = False
            for pred in pred_ranked[:k]:
                if len(tset & set(pred)) >= 0.8 * len(tset):
                    found = True
                    break
            hits += found
        out[f"top{k}_recovery"] = hits / max(1, len(true_sequences))
    return out


def ranking_metrics(relevances: List[List[float]]) -> Dict[str, float]:
    """MRR + NDCG over per-query ranked relevance lists (doc §35)."""
    def dcg(rels: List[float]) -> float:
        return sum(r / np.log2(i + 2) for i, r in enumerate(rels))

    mrr, ndcgs = [], []
    for rels in relevances:
        if not rels or max(rels) <= 0:
            continue
        best_rank = next((i + 1 for i, r in enumerate(rels) if r > 0), None)
        mrr.append(1.0 / best_rank if best_rank else 0.0)
        ideal = sorted(rels, reverse=True)
        d, di = dcg(rels), dcg(ideal)
        ndcgs.append(d / di if di > 0 else 0.0)
    return {"mrr": float(np.mean(mrr)) if mrr else 0.0,
            "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0}


def anomaly_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred.astype(int),
                                      labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def roc_curve_points(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, list]:
    fpr, tpr, _ = roc_curve(y_true.astype(int), proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def pr_curve_points(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, list]:
    from sklearn.metrics import precision_recall_curve

    prec, rec, _ = precision_recall_curve(y_true.astype(int), proba)
    return {"precision": prec.tolist(), "recall": rec.tolist()}
