"""Sliding-window / local entropy features.

Computes entropy over small byte windows inside each fragment to expose
*local* structure that a whole-block histogram hides: headers/footers are
low-entropy islands inside high-entropy payload, format transitions show up
as entropy gradients, and corrupted regions break the local profile.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

WINDOW = 64
STRIDE = 32
PROFILE_DIM = 15  # fixed-length downsampled entropy profile per fragment


def entropy_of_window(w: np.ndarray) -> float:
    if w.size == 0:
        return 0.0
    counts = np.bincount(w, minlength=256).astype(np.float64)
    p = counts[counts > 0] / w.size
    return float(-(p * np.log2(p)).sum())


def local_entropy_profile(data: np.ndarray, window: int = WINDOW,
                          stride: int = STRIDE) -> np.ndarray:
    """Variable-length local entropy profile for one fragment."""
    if data.size < window:
        return np.array([entropy_of_window(data)])
    positions = range(0, data.size - window + 1, stride)
    return np.array([entropy_of_window(data[i:i + window]) for i in positions])


def local_entropy_features(data: np.ndarray) -> Dict[str, float]:
    prof = local_entropy_profile(data)
    if prof.size == 0:
        prof = np.zeros(1)
    mean = float(prof.mean())
    std = float(prof.std())
    peak = prof > mean + std
    # normalised position of the maximum-entropy window (structure signature)
    max_pos = float(np.argmax(prof)) / max(1, prof.size - 1)
    grad = float(prof[-1] - prof[0])
    return {
        "le_mean": mean,
        "le_std": std,
        "le_min": float(prof.min()),
        "le_max": float(prof.max()),
        "le_range": float(prof.max() - prof.min()),
        "le_gradient": grad,
        "le_peak_ratio": float(peak.mean()),
        "le_max_pos": max_pos,
        "le_head": float(prof[0]),
        "le_tail": float(prof[-1]),
        "le_low_entropy_fraction": float((prof < 3.0).mean()),
        "le_profile": _resize_profile(prof),
    }


def _resize_profile(prof: np.ndarray) -> np.ndarray:
    """Resample the entropy profile to a fixed length (PROFILE_DIM)."""
    if prof.size == PROFILE_DIM:
        return prof.astype(np.float64)
    if prof.size == 0:
        return np.zeros(PROFILE_DIM)
    x_old = np.linspace(0.0, 1.0, prof.size)
    x_new = np.linspace(0.0, 1.0, PROFILE_DIM)
    return np.interp(x_new, x_old, prof)


def local_entropy_matrix(data_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised over an (n, block) matrix."""
    n = data_matrix.shape[0]
    scalar_keys = ("le_mean", "le_std", "le_min", "le_max", "le_range", "le_gradient",
                   "le_peak_ratio", "le_max_pos", "le_head", "le_tail",
                   "le_low_entropy_fraction")
    out = {k: np.zeros(n) for k in scalar_keys}
    profiles = np.zeros((n, PROFILE_DIM))
    for i in range(n):
        f = local_entropy_features(data_matrix[i])
        for k in scalar_keys:
            out[k][i] = f[k]
        profiles[i] = f["le_profile"]
    out["le_profile"] = profiles
    return out


def local_entropy_pair_features(prof_a: np.ndarray, prof_b: np.ndarray,
                                tail_a: float, head_b: float) -> Dict[str, float]:
    """Pairwise local-entropy evidence (family: statistical)."""
    na, nb = np.linalg.norm(prof_a), np.linalg.norm(prof_b)
    cos = float(np.dot(prof_a, prof_b) / (na * nb)) if na > 0 and nb > 0 else 0.0
    return {
        "le_profile_cosine": cos,
        "le_boundary_diff": abs(tail_a - head_b),
        "le_boundary_continuity": float(np.clip(1.0 - abs(tail_a - head_b) / 4.0, 0.0, 1.0)),
        "le_similar_structure": float(abs(prof_a.mean() - prof_b.mean()) < 0.75),
    }
