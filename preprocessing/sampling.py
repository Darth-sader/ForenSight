"""Pair/label generation and class balancing (doc §8, §11).

Positive/negative pair generation for the two supervised tasks:
  Task 1 - same-file classification      P(A, B in same file)
  Task 2 - next-fragment prediction      P(B immediately follows A)
Class balancing uses stratified sampling, class weights and controlled
negative sampling (easy + hard negatives); SMOTE is provided for linear
models where scientifically justified, but is OFF by default - synthetic
oversampling of raw forensic bytes is avoided (doc §8).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PairDataset:
    pair_idx: np.ndarray          # (n, 2) fragment row indices
    X: np.ndarray                 # (n, d) pair features
    y_same: np.ndarray            # (n,) task 1 labels
    y_next: np.ndarray            # (n,) task 2 labels
    hard_negative: np.ndarray     # (n,) bool
    group: np.ndarray             # (n,) source-file group id of fragment A
    feature_names: List[str]

    def __len__(self) -> int:
        return len(self.y_same)

    def __getitem__(self, mask: np.ndarray) -> "PairDataset":
        return PairDataset(pair_idx=self.pair_idx[mask], X=self.X[mask],
                           y_same=self.y_same[mask], y_next=self.y_next[mask],
                           hard_negative=self.hard_negative[mask],
                           group=self.group[mask], feature_names=self.feature_names)


def class_weights(y: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(y.astype(int))
    total = len(y)
    return {c: float(total / (len(counts) * max(1, counts[c]))) for c in range(len(counts))}


def smote_numeric(X: np.ndarray, y: np.ndarray, target_class: int = 1,
                  k: int = 5, seed: int = 42, max_synth: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """Textbook SMOTE for numeric feature vectors (doc §8: where justified).

    Returns the original arrays augmented with synthetic minority samples.
    """
    rng = np.random.RandomState(seed)
    minority = np.where(y == target_class)[0]
    majority_n = int((y != target_class).sum())
    deficit = majority_n - len(minority)
    if deficit <= 0 or len(minority) < k + 1:
        return X, y
    X_min = X[minority]
    synth = np.zeros((min(deficit, max_synth), X.shape[1]), dtype=X.dtype)
    for s in range(synth.shape[0]):
        i = rng.randint(len(X_min))
        nn = rng.choice(len(X_min), size=k, replace=False)
        j = nn[0] if nn[0] != i else nn[1]
        lam = rng.rand()
        synth[s] = X_min[i] + lam * (X_min[j] - X_min[i])
    return np.vstack([X, synth]), np.concatenate([y, np.full(len(synth), target_class, dtype=y.dtype)])


def generate_pairs(metas: List[Dict[str, object]], fragment_rows: Sequence[int],
                   neg_ratio: float = 2.0, same_ratio: float = 1.0,
                   hard_negative_frac: float = 0.5, seed: int = 42,
                   restrict_same_split: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build labelled pair sets from fragment metadata.

    ``metas`` rows need: source_file, original_position, file_type, split.
    ``fragment_rows`` maps meta positions to FeatureEngine row indices.

    Returns (pair_idx, y_same, y_next, hard_negative, group).
    """
    rng = random.Random(seed)
    by_file: Dict[str, List[int]] = {}
    by_type: Dict[str, List[int]] = {}
    for local_i, row in enumerate(fragment_rows):
        m = metas[row]
        by_file.setdefault(str(m["source_file"]), []).append(local_i)
        by_type.setdefault(str(m["file_type"]), []).append(local_i)

    pairs: List[Tuple[int, int]] = []
    y_same: List[int] = []
    y_next: List[int] = []
    hard: List[bool] = []
    groups: List[str] = []
    files = list(by_file.keys())

    for fname, members in by_file.items():
        members.sort(key=lambda li: int(metas[fragment_rows[li]]["original_position"]))
        n_same_added = 0
        for a in range(len(members) - 1):
            i, j = members[a], members[a + 1]
            pairs.append((i, j)); y_same.append(1); y_next.append(1)
            hard.append(False); groups.append(fname)
            # same-file but non-adjacent positives (task 1 only)
            if same_ratio > 0 and len(members) > 2 and n_same_added < same_ratio * (len(members) - 1):
                k = rng.choice([x for x in members if x not in (i, j)])
                if abs(members.index(k) - a) > 1:
                    pairs.append((i, k)); y_same.append(1); y_next.append(0)
                    hard.append(False); groups.append(fname)
                    n_same_added += 1

    n_pos = len(pairs)
    n_neg = int(neg_ratio * n_pos)
    n_hard = int(hard_negative_frac * n_neg)
    # hard negatives: same file type, different file (incl. similar-content groups)
    for _ in range(n_hard):
        f = rng.choice(files)
        others = [x for x in files if x != f and
                  metas[fragment_rows[by_file[f][0]]]["file_type"] ==
                  metas[fragment_rows[by_file[x][0]]]["file_type"]]
        if not others:
            continue
        g = rng.choice(others)
        i, j = rng.choice(by_file[f]), rng.choice(by_file[g])
        pairs.append((i, j)); y_same.append(0); y_next.append(0)
        hard.append(True); groups.append(str(metas[fragment_rows[i]]["source_file"]))
    # easy negatives: random cross-type pairs
    while len(pairs) < n_pos + n_neg:
        i, j = rng.choice(range(len(fragment_rows))), rng.choice(range(len(fragment_rows)))
        fi = str(metas[fragment_rows[i]]["source_file"])
        fj = str(metas[fragment_rows[j]]["source_file"])
        if fi == fj:
            continue
        pairs.append((i, j)); y_same.append(0); y_next.append(0)
        hard.append(False); groups.append(fi)

    order = list(range(len(pairs)))
    rng.shuffle(order)
    pair_idx = np.array([pairs[o] for o in order], dtype=np.int64)
    return (pair_idx, np.array([y_same[o] for o in order]),
            np.array([y_next[o] for o in order]),
            np.array([hard[o] for o in order], dtype=bool),
            np.array([hash(groups[o]) & 0xFFFFFFFF for o in order], dtype=np.int64))
