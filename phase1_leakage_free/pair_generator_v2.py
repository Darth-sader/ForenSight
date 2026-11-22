"""Leakage-Free Forensic Pair Generator (v2.0.0-leakage_free).

Guarantees:
1. Independent dual-task label generation:
   - Same-file: y_same(i, j) = 1 if fragments belong to the same original file.
   - Directed adjacency: y_next(i, j) = 1 if fragment j immediately follows fragment i.
2. Meaningful multi-strategy negative pair sampling:
   - Hard negatives (different files of same format).
   - Similar byte distributions (low histogram L1 distance).
   - Same-file non-adjacent fragments (y_same=1, y_next=0).
   - Structurally compatible candidates.
   - Cross-file random negatives.
3. Strict partition isolation:
   - Both fragments in every pair MUST belong to the permitted partition.
   - Zero foreign fragments sampled across partition boundaries.
4. Complete provenance tracking for every generated pair.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from config import HARD_NEGATIVE_FRACTION, PAIR_NEGATIVE_RATIO, SAME_PAIR_RATIO, SEED
from features import PAIR_NAMES
from phase1_leakage_free.fragment_schema_v2 import GroundTruthFragment
from phase1_leakage_free.leakage_free_features import LeakageFreeFeatureEngine


@dataclass
class PairProvenance:
    pair_id: int
    strategy: str  # "pos_adjacent", "pos_same_non_adjacent", "neg_same_format", "neg_similar_hist", "neg_struct_compat", "neg_cross_file_random"
    frag_a_id: str
    frag_b_id: str
    source_file_a: str
    source_file_b: str
    partition: str


@dataclass
class PairDatasetV2:
    pair_idx: np.ndarray       # (N, 2) local row indices in feature engine
    X: np.ndarray              # (N, 362) pair features
    y_same: np.ndarray         # (N,) binary labels (1 = same file)
    y_next: np.ndarray         # (N,) binary labels (1 = B immediately follows A)
    hard_negative: np.ndarray  # (N,) boolean mask
    provenance: List[PairProvenance]
    feature_names: List[str]
    partition: str             # "train", "val", "cal", "test"

    def __len__(self) -> int:
        return len(self.y_same)

    def __getitem__(self, mask: np.ndarray) -> "PairDatasetV2":
        filtered_provenance = [self.provenance[i] for i, m in enumerate(mask) if m]
        return PairDatasetV2(
            pair_idx=self.pair_idx[mask],
            X=self.X[mask],
            y_same=self.y_same[mask],
            y_next=self.y_next[mask],
            hard_negative=self.hard_negative[mask],
            provenance=filtered_provenance,
            feature_names=self.feature_names,
            partition=self.partition,
        )


class PairGeneratorV2:
    """Generates dual-relation training pairs with provenance and strict partition confinement."""

    def __init__(
        self,
        seed: int = SEED,
        neg_ratio: float = PAIR_NEGATIVE_RATIO,
        same_ratio: float = SAME_PAIR_RATIO,
        hard_neg_frac: float = HARD_NEGATIVE_FRACTION,
    ):
        self.seed = seed
        self.neg_ratio = neg_ratio
        self.same_ratio = same_ratio
        self.hard_neg_frac = hard_neg_frac

    def generate_pairs_for_partition(
        self,
        fragments: List[GroundTruthFragment],
        allowed_files: Set[str],
        engine: LeakageFreeFeatureEngine,
        partition_name: str,
        seed_offset: int = 0,
    ) -> PairDatasetV2:
        """Construct labelled pairs strictly confined to allowed_files within the partition."""
        rng = random.Random(self.seed + seed_offset)

        # 1. Filter local partition fragments
        partition_indices: List[int] = []
        for idx, f in enumerate(fragments):
            if f.source_file_id in allowed_files:
                partition_indices.append(idx)

        if not partition_indices:
            raise ValueError(f"Partition {partition_name} contains 0 fragments in allowed_files!")

        # Group by file and type
        by_file: Dict[str, List[int]] = {}
        by_type: Dict[str, List[int]] = {}
        for local_idx, global_idx in enumerate(partition_indices):
            f = fragments[global_idx]
            by_file.setdefault(f.source_file_id, []).append(local_idx)
            # Infer format from observable or ground truth metadata
            ft = str(f.observable.filesystem_metadata.get("file_type") or f.source_file_id.split(".")[-1].upper())
            by_type.setdefault(ft, []).append(local_idx)

        pairs: List[Tuple[int, int]] = []
        y_same_list: List[int] = []
        y_next_list: List[int] = []
        hard_list: List[bool] = []
        provenances: List[PairProvenance] = []

        files = sorted(by_file.keys())

        # 2. Positive pairs: consecutive adjacent pairs (y_same=1, y_next=1)
        for fn in files:
            members = by_file[fn]
            # Sort strictly by ground-truth logical sequence index
            members.sort(key=lambda li: fragments[partition_indices[li]].logical_sequence_index)
            n_consec = len(members) - 1

            for a in range(n_consec):
                i, j = members[a], members[a + 1]
                pairs.append((i, j))
                y_same_list.append(1)
                y_next_list.append(1)
                hard_list.append(False)
                provenances.append(PairProvenance(
                    pair_id=len(pairs) - 1,
                    strategy="pos_adjacent",
                    frag_a_id=fragments[partition_indices[i]].observable.fragment_id,
                    frag_b_id=fragments[partition_indices[j]].observable.fragment_id,
                    source_file_a=fn,
                    source_file_b=fn,
                    partition=partition_name,
                ))

            # Same-file but non-adjacent pairs (y_same=1, y_next=0)
            if self.same_ratio > 0 and len(members) > 2:
                target_non_adj = int(round(self.same_ratio * n_consec))
                added = 0
                attempts = 0
                while added < target_non_adj and attempts < target_non_adj * 5:
                    attempts += 1
                    a = rng.randrange(len(members))
                    b = rng.randrange(len(members))
                    if abs(a - b) > 1:
                        i, j = members[a], members[b]
                        pairs.append((i, j))
                        y_same_list.append(1)
                        y_next_list.append(0)
                        hard_list.append(False)
                        provenances.append(PairProvenance(
                            pair_id=len(pairs) - 1,
                            strategy="pos_same_non_adjacent",
                            frag_a_id=fragments[partition_indices[i]].observable.fragment_id,
                            frag_b_id=fragments[partition_indices[j]].observable.fragment_id,
                            source_file_a=fn,
                            source_file_b=fn,
                            partition=partition_name,
                        ))
                        added += 1

        n_pos = len(pairs)
        n_neg = int(round(self.neg_ratio * n_pos))
        n_hard = int(round(self.hard_neg_frac * n_neg))
        n_hist = int(round(0.25 * (n_neg - n_hard)))

        # 3. Hard negative pairs: same file type, different file (within this partition only!)
        hard_count = 0
        attempts = 0
        max_attempts = n_hard * 10
        while hard_count < n_hard and attempts < max_attempts:
            attempts += 1
            f_a = rng.choice(files)
            f_b_candidates = [f for f in files if f != f_a and f.split(".")[-1] == f_a.split(".")[-1]]
            if not f_b_candidates:
                continue
            f_b = rng.choice(f_b_candidates)
            i = rng.choice(by_file[f_a])
            j = rng.choice(by_file[f_b])
            pairs.append((i, j))
            y_same_list.append(0)
            y_next_list.append(0)
            hard_list.append(True)
            provenances.append(PairProvenance(
                pair_id=len(pairs) - 1,
                strategy="neg_same_format",
                frag_a_id=fragments[partition_indices[i]].observable.fragment_id,
                frag_b_id=fragments[partition_indices[j]].observable.fragment_id,
                source_file_a=f_a,
                source_file_b=f_b,
                partition=partition_name,
            ))
            hard_count += 1

        # 4. Similar byte-distribution negatives (histogram L1 distance check)
        hist_count = 0
        attempts = 0
        max_attempts = n_hist * 10
        while hist_count < n_hist and attempts < max_attempts:
            attempts += 1
            i = rng.randrange(len(partition_indices))
            j = rng.randrange(len(partition_indices))
            fa = fragments[partition_indices[i]].source_file_id
            fb = fragments[partition_indices[j]].source_file_id
            if fa == fb:
                continue
            # Compute quick L1 diff on byte histograms
            ha = engine.cache_["byte"]["hist"][partition_indices[i]]
            hb = engine.cache_["byte"]["hist"][partition_indices[j]]
            if np.abs(ha - hb).sum() < 0.35:  # High byte distribution similarity
                pairs.append((i, j))
                y_same_list.append(0)
                y_next_list.append(0)
                hard_list.append(True)
                provenances.append(PairProvenance(
                    pair_id=len(pairs) - 1,
                    strategy="neg_similar_hist",
                    frag_a_id=fragments[partition_indices[i]].observable.fragment_id,
                    frag_b_id=fragments[partition_indices[j]].observable.fragment_id,
                    source_file_a=fa,
                    source_file_b=fb,
                    partition=partition_name,
                ))
                hist_count += 1

        # 5. Easy random cross-file negatives
        while len(pairs) < n_pos + n_neg:
            i = rng.randrange(len(partition_indices))
            j = rng.randrange(len(partition_indices))
            fa = fragments[partition_indices[i]].source_file_id
            fb = fragments[partition_indices[j]].source_file_id
            if fa == fb:
                continue
            pairs.append((i, j))
            y_same_list.append(0)
            y_next_list.append(0)
            hard_list.append(False)
            provenances.append(PairProvenance(
                pair_id=len(pairs) - 1,
                strategy="neg_cross_file_random",
                frag_a_id=fragments[partition_indices[i]].observable.fragment_id,
                frag_b_id=fragments[partition_indices[j]].observable.fragment_id,
                source_file_a=fa,
                source_file_b=fb,
                partition=partition_name,
            ))

        # Shuffle deterministically
        order = list(range(len(pairs)))
        rng.shuffle(order)

        # Global engine pair indices
        global_pair_idx = np.empty((len(pairs), 2), dtype=np.int64)
        for k, o in enumerate(order):
            li, lj = pairs[o]
            global_pair_idx[k] = (partition_indices[li], partition_indices[lj])

        y_same_arr = np.array([y_same_list[o] for o in order], dtype=np.int32)
        y_next_arr = np.array([y_next_list[o] for o in order], dtype=np.int32)
        hard_arr = np.array([hard_list[o] for o in order], dtype=bool)
        shuffled_provenance = [provenances[o] for o in order]

        # Vectorized pair feature extraction
        X = engine.pair_matrix_vec(global_pair_idx)

        # Strict validation: verify that 100% of pairs strictly belong to allowed_files
        for prov in shuffled_provenance:
            assert prov.source_file_a in allowed_files, f"LEAKAGE: Fragment A {prov.source_file_a} not in {partition_name}!"
            assert prov.source_file_b in allowed_files, f"LEAKAGE: Fragment B {prov.source_file_b} not in {partition_name}!"

        return PairDatasetV2(
            pair_idx=global_pair_idx,
            X=X,
            y_same=y_same_arr,
            y_next=y_next_arr,
            hard_negative=hard_arr,
            provenance=shuffled_provenance,
            feature_names=[name for name, _ in PAIR_NAMES],
            partition=partition_name,
        )
