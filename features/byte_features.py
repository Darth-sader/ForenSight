"""Byte-level features (doc §9A): 256-d histogram, mean, variance, min, max, entropy."""
from __future__ import annotations

from typing import Dict

import numpy as np

BYTE_HIST_DIM = 256


def compute_byte_features(data: np.ndarray) -> Dict[str, np.ndarray | float]:
    """``data`` is a 1-D uint8 array of raw fragment bytes."""
    counts = np.bincount(data, minlength=256).astype(np.float64)
    hist = counts / max(1, data.size)
    non_zero = counts[counts > 0]
    probs = non_zero / data.size
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
    return {
        "hist": hist,
        "mean": float(data.mean()) if data.size else 0.0,
        "variance": float(data.var()) if data.size else 0.0,
        "min": float(data.min()) if data.size else 0.0,
        "max": float(data.max()) if data.size else 0.0,
        "entropy": entropy,
    }


def compute_byte_matrix(data_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised computation over an (n, block) uint8 matrix.

    Returns dict with keys ``hist`` (n,256) plus one (n,) array per scalar.
    """
    n = data_matrix.shape[0]
    hist = np.zeros((n, BYTE_HIST_DIM), dtype=np.float64)
    for i in range(n):
        hist[i] = np.bincount(data_matrix[i], minlength=256) / data_matrix.shape[1]
    entropy = -np.sum(np.where(hist > 0, hist * np.log2(np.maximum(hist, 1e-12)), 0.0), axis=1)
    return {
        "hist": hist,
        "mean": data_matrix.mean(axis=1),
        "variance": data_matrix.var(axis=1),
        "min": data_matrix.min(axis=1).astype(np.float64),
        "max": data_matrix.max(axis=1).astype(np.float64),
        "entropy": entropy,
    }
