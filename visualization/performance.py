"""Model performance visualisations (doc §37): ROC/PR curves, confusion
matrices, calibration curves, feature importance, SHAP summary, sweep lines
and candidate confidence comparisons."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR


def plot_roc_curves(curves: Dict[str, Dict[str, list]], name: str = "roc_curves") -> str:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, c in curves.items():
        ax.plot(c["fpr"], c["tpr"], lw=1.6, label=label)
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - next-fragment prediction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)


def plot_pr_curves(curves: Dict[str, Dict[str, list]], name: str = "pr_curves") -> str:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, c in curves.items():
        ax.plot(c["recall"], c["precision"], lw=1.6, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)


def plot_confusion_matrices(mats: Dict[str, Dict[str, int]],
                            name: str = "confusion_matrices") -> str:
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
    axes = np.atleast_1d(axes)
    for ax, (label, m) in zip(axes, mats.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(label, fontsize=9)
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)


def plot_calibration(reliability: Dict[str, Dict[str, list]], name: str = "calibration") -> str:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, r in reliability.items():
        ax.plot(r["confidence"], r["accuracy"], marker="o", ms=4, lw=1.4, label=label)
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Calibration (reliability) curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)


def plot_feature_importance(importances: List[Tuple[str, float]],
                            name: str = "feature_importance", top_k: int = 20) -> str:
    top = importances[:top_k][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(top) + 1.2))
    ax.barh([t[0] for t in top], [t[1] for t in top], color="#4c72b0")
    ax.set_xlabel("Importance")
    ax.set_title("Top features (model importance)")
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)


def plot_shap_summary(shap_matrix: np.ndarray, feature_names: List[str],
                      X: np.ndarray, name: str = "shap_summary", top_k: int = 12) -> str:
    if shap_matrix.size == 0 or shap_matrix.ndim != 2:
        return ""
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_k][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].barh([feature_names[j] for j in order], mean_abs[order], color="#55a868")
    axes[0].set_xlabel("mean |SHAP|")
    axes[0].set_title("Global SHAP importance")
    j = order[-1]
    axes[1].scatter(X[:200, j], shap_matrix[:200, j], s=8, alpha=0.6, color="#c44e52")
    axes[1].set_xlabel(f"{feature_names[j]} value")
    axes[1].set_ylabel("SHAP value")
    axes[1].set_title(f"Dependence: {feature_names[j]}")
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)


def plot_sweep(x_labels: Sequence[str], series: Dict[str, Sequence[float]],
               xlabel: str, ylabel: str, title: str, name: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, ys in series.items():
        ax.plot(x_labels, ys, marker="o", lw=1.6, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)


def plot_candidate_confidence(candidates: List[Dict[str, object]],
                              name: str = "candidate_confidence") -> str:
    """Bar chart of calibrated confidence per candidate with classification colour."""
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(candidates)), 4.6))
    ids = [c["candidate_id"] for c in candidates]
    conf = [float(c["confidence"]) for c in candidates]
    colors = ["#55a868" if c["classification"] == "HIGH-CONFIDENCE RECOVERY"
              else "#dd8452" if c["classification"] == "UNCERTAIN" else "#c44e52"
              for c in candidates]
    ax.bar(ids, conf, color=colors)
    ax.axhline(0.85, ls="--", color="#55a868", lw=1, label="high-confidence threshold")
    ax.axhline(0.60, ls="--", color="#dd8452", lw=1, label="uncertain threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Calibrated confidence")
    ax.set_title("Ranked reconstruction candidates")
    ax.tick_params(axis="x", labelrotation=60, labelsize=6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = PLOTS_DIR / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)
