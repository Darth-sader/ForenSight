"""Statistical features (doc §9A): printable/ASCII/zero/high-byte ratios, byte
diversity, hashed n-gram frequency vectors."""
from __future__ import annotations

from typing import Dict

import numpy as np

NGRAM_DIM = 32          # hashed 2-gram and 3-gram vector sizes
ASCII_LO, ASCII_HI = 0x20, 0x7E


def _hash_index(b0: int, b1: int = -1, b2: int = -1) -> int:
    h = (b0 * 2654435761 + b1 * 40503 + b2 * 65599) & 0xFFFFFFFF
    return h % NGRAM_DIM


def ngram_vector(data: np.ndarray, n: int) -> np.ndarray:
    vec = np.zeros(NGRAM_DIM, dtype=np.float64)
    if data.size < n:
        return vec
    norm = 0.0
    for i in range(data.size - n + 1):
        if n == 2:
            vec[_hash_index(int(data[i]), int(data[i + 1]))] += 1.0
        else:
            vec[_hash_index(int(data[i]), int(data[i + 1]), int(data[i + 2]))] += 1.0
        norm += 1.0
    if norm > 0:
        vec /= norm
    return vec


def compute_statistical_features(data: np.ndarray, byte_feats: Dict[str, np.ndarray | float]) -> Dict[str, np.ndarray | float]:
    printable = ((data >= ASCII_LO) & (data <= ASCII_HI)).mean()
    ascii_ext = ((data >= ASCII_LO) & (data <= ASCII_HI)) | (data < 0x20)
    counts = np.bincount(data, minlength=256)
    unique = int((counts > 0).sum())
    top = counts.max() / max(1, data.size)
    return {
        "printable_ratio": float(printable),
        "ascii_ratio": float(ascii_ext.mean()),
        "zero_ratio": float((data == 0).mean()),
        "high_byte_ratio": float((data >= 0x80).mean()),
        "byte_diversity": unique / 256.0,
        "top_byte_freq": float(top),
        "repeat_run_ratio": _repeat_run_ratio(data),
        "ngram2": ngram_vector(data, 2),
        "ngram3": ngram_vector(data, 3),
        "bigram_entropy": _vec_entropy(ngram_vector(data, 2)),
    }


def _repeat_run_ratio(data: np.ndarray) -> float:
    """Fraction of positions that continue the previous byte value (run length)."""
    if data.size < 2:
        return 0.0
    return float(np.mean(data[1:] == data[:-1]))


def _vec_entropy(vec: np.ndarray) -> float:
    p = vec[vec > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def compute_statistical_matrix(data_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    n = data_matrix.shape[0]
    out = {k: np.zeros(n) for k in ("printable_ratio", "ascii_ratio", "zero_ratio",
                                    "high_byte_ratio", "byte_diversity", "top_byte_freq",
                                    "repeat_run_ratio", "bigram_entropy")}
    ngram2 = np.zeros((n, NGRAM_DIM))
    ngram3 = np.zeros((n, NGRAM_DIM))
    for i in range(n):
        f = compute_statistical_features(data_matrix[i], {})
        for k in out:
            out[k][i] = f[k]
        ngram2[i] = f["ngram2"]
        ngram3[i] = f["ngram3"]
    out["ngram2"] = ngram2
    out["ngram3"] = ngram3
    return out
