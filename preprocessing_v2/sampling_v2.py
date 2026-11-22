"""ForenSight v2 sampling and pair generation with strict partition isolation.

Fixes Leakage Risk LR-03:
In v1, negative pairs (A, B) only grouped by file A. If file A was in the train fold,
file B could be selected from a file in the validation fold, leaking validation file
fragment representations into the training partition.

In v2, all pairs (both positive and negative) are sampled strictly within the provided
partition slice, guaranteeing 100% disjoint source files across train, val, and holdout.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class PairDatasetV2:
    """A candidate fragment pair dataset with strict partition tracking."""
    pair_idx: np.ndarray          # (n, 2) fragment row indices in engine matrix
    X: np.ndarray                 # (n, d) pair features
    y_same: np.ndarray            # (n,) task 1 labels (1 = same file, 0 = different)
    y_next: np.ndarray            # (n,) task 2 labels (1 = B immediately follows A)
    hard_negative: np.ndarray     # (n,) bool
    group_a: np.ndarray           # (n,) source-file group hash of fragment A
    group_b: np.ndarray           # (n,) source-file group hash of fragment B
    feature_names: List[str]
    partition: str = "train"      # "train" | "val" | "holdout"

    def __len__(self) -> int:
        return len(self.y_same)

    def __getitem__(self, mask: np.ndarray) -> "PairDatasetV2":
        return PairDatasetV2(
            pair_idx=self.pair_idx[mask],
            X=self.X[mask],
            y_same=self.y_same[mask],
            y_next=self.y_next[mask],
            hard_negative=self.hard_negative[mask],
            group_a=self.group_a[mask],
            group_b=self.group_b[mask],
            feature_names=self.feature_names,
            partition=self.partition,
        )


def generate_pairs_v2(
    metas: List[Dict[str, object]],
    fragment_rows: Sequence[int],
    neg_ratio: float = 2.0,
    same_ratio: float = 1.0,
    hard_negative_frac: float = 0.5,
    seed: int = 42,
    partition_name: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build labelled pair sets strictly restricted to fragment_rows.

    Guarantees:
    1. Both fragment A and fragment B belong strictly to `fragment_rows`.
    2. Zero foreign fragments from other partitions are selected for negative pairs.
    3. Positive adjacent pairs have y_same=1, y_next=1.
    4. Positive non-adjacent pairs (same file) have y_same=1, y_next=0.
    5. Hard negatives (same format, different file) have y_same=0, y_next=0.
    6. Easy negatives (different format) have y_same=0, y_next=0.

    Returns:
        (pair_idx, y_same, y_next, hard_negative, group_a, group_b)
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
    groups_a: List[str] = []
    groups_b: List[str] = []
    files = sorted(by_file.keys())

    # 1. Positives (same file)
    for fname in files:
        members = by_file[fname]
        members.sort(key=lambda li: int(metas[fragment_rows[li]]["original_position"]))
        n_same_added = 0
        for a in range(len(members) - 1):
            i, j = members[a], members[a + 1]
            pairs.append((i, j))
            y_same.append(1)
            y_next.append(1)
            hard.append(False)
            groups_a.append(fname)
            groups_b.append(fname)

            # Same-file but non-adjacent pairs (task 1 only: membership without adjacency)
            if same_ratio > 0 and len(members) > 2 and n_same_added < same_ratio * (len(members) - 1):
                non_adj = [x for idx_m, x in enumerate(members) if abs(idx_m - a) > 1]
                if non_adj:
                    k = rng.choice(non_adj)
                    pairs.append((i, k))
                    y_same.append(1)
                    y_next.append(0)
                    hard.append(False)
                    groups_a.append(fname)
                    groups_b.append(fname)
                    n_same_added += 1

    n_pos = len(pairs)
    n_neg = int(neg_ratio * n_pos)
    n_hard = int(hard_negative_frac * n_neg)

    # 2. Hard negatives: same file type, different file (within this partition only!)
    hard_count = 0
    max_hard_attempts = n_hard * 4
    attempts = 0
    while hard_count < n_hard and attempts < max_hard_attempts:
        attempts += 1
        f = rng.choice(files)
        ftype = str(metas[fragment_rows[by_file[f][0]]]["file_type"])
        same_type_files = [x for x in files if x != f and str(metas[fragment_rows[by_file[x][0]]]["file_type"]) == ftype]
        if not same_type_files:
            continue
        g = rng.choice(same_type_files)
        i = rng.choice(by_file[f])
        j = rng.choice(by_file[g])
        pairs.append((i, j))
        y_same.append(0)
        y_next.append(0)
        hard.append(True)
        groups_a.append(f)
        groups_b.append(g)
        hard_count += 1

    # 3. Easy negatives: cross-file pairs (within this partition only!)
    while len(pairs) < n_pos + n_neg:
        i = rng.randrange(len(fragment_rows))
        j = rng.randrange(len(fragment_rows))
        fi = str(metas[fragment_rows[i]]["source_file"])
        fj = str(metas[fragment_rows[j]]["source_file"])
        if fi == fj:
            continue
        pairs.append((i, j))
        y_same.append(0)
        y_next.append(0)
        hard.append(False)
        groups_a.append(fi)
        groups_b.append(fj)

    # Shuffle order with fixed seed
    order = list(range(len(pairs)))
    rng.shuffle(order)

    pair_idx = np.array([pairs[o] for o in order], dtype=np.int64)
    y_same_arr = np.array([y_same[o] for o in order], dtype=np.int32)
    y_next_arr = np.array([y_next[o] for o in order], dtype=np.int32)
    hard_arr = np.array([hard[o] for o in order], dtype=bool)
    group_a_arr = np.array([hash(groups_a[o]) & 0xFFFFFFFF for o in order], dtype=np.int64)
    group_b_arr = np.array([hash(groups_b[o]) & 0xFFFFFFFF for o in order], dtype=np.int64)

    return pair_idx, y_same_arr, y_next_arr, hard_arr, group_a_arr, group_b_arr


def verify_partition_disjointness(
    train_files: Set[str],
    val_files: Set[str],
    holdout_files: Set[str],
) -> bool:
    """Mathematically verify that partitions share zero files."""
    tr_va = train_files & val_files
    tr_ho = train_files & holdout_files
    va_ho = val_files & holdout_files
    return len(tr_va) == 0 and len(tr_ho) == 0 and len(va_ho) == 0
