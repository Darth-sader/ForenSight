"""ForenSight v2 label generation and corpus pair-dataset assembly.

Fixes Critical Leakage Risks:
- LR-01: Context metadata leakage. In v1, physical_offset = offset_in_source and
  cluster = original_position, giving positive pairs physical_adjacent == 1.0.
  v2 defaults to context_mode="metadata_independent", zeroing offsets.
- LR-02: Normalization leakage. v2 fits FeatureNormalizerV2 strictly on the training
  partition and transforms validation and holdout out-of-sample.
- LR-03: Asymmetric negative pair leakage. v2 performs a strict 3-way partition
  (train_files, val_files, holdout_files) and samples pairs strictly within partition.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from common import logger, set_seeds, timer
from config import (CORRUPTION_LEVELS, FRAGMENT_SIZE, FRAGMENTS_DIR,
                    GROUND_TRUTH_DIR, HARD_NEGATIVE_FRACTION, PAIR_NEGATIVE_RATIO,
                    SAME_PAIR_RATIO, SEED)
from features_v2 import FeatureEngineV2, PAIR_NAMES
from preprocessing_v2.normalize_v2 import FeatureNormalizerV2
from preprocessing_v2.sampling_v2 import (PairDatasetV2, generate_pairs_v2,
                                          verify_partition_disjointness)


def _pad(buf: bytes, size: int) -> bytes:
    return buf if len(buf) >= size else buf.ljust(size, b"\x00")


class CorpusPairBuilderV2:
    """Builds strictly isolated train, validation, and holdout pair datasets."""

    def __init__(
        self,
        seed: int = SEED,
        context_mode: str = "metadata_independent",
        corruption_aug_frac: float = 0.35,
        neg_ratio: float = PAIR_NEGATIVE_RATIO,
        same_ratio: float = SAME_PAIR_RATIO,
        val_file_ratio: float = 0.15,
        normalize_method: str = "standard",
    ):
        if context_mode not in ("metadata_independent", "simulated_layout", "ground_truth_debug"):
            raise ValueError(f"Unknown context_mode: {context_mode}")

        self.seed = seed
        self.rng = random.Random(seed)
        self.context_mode = context_mode
        self.corruption_aug_frac = corruption_aug_frac
        self.neg_ratio = neg_ratio
        self.same_ratio = same_ratio
        self.val_file_ratio = val_file_ratio
        self.normalize_method = normalize_method

    def load_corpus(self) -> Tuple[List[Dict[str, str]], List[bytes]]:
        """Load ground truth metadata rows and binary fragment blobs."""
        with open(GROUND_TRUTH_DIR / "fragments.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        rows.sort(key=lambda r: int(r["fragment_id"][1:]))
        blobs = [(FRAGMENTS_DIR / "blobs" / f"{r['fragment_id']}.bin").read_bytes() for r in rows]
        return rows, blobs

    def _partition_files(
        self, metas_raw: List[Dict[str, str]]
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Partition source files into mutually disjoint train, val, and holdout sets."""
        by_split_and_type: Dict[str, Dict[str, List[str]]] = {}
        file_to_type: Dict[str, str] = {}

        for r in metas_raw:
            fn = r["source_file"]
            sp = r["split"]
            ft = r["file_type"]
            file_to_type[fn] = ft
            by_split_and_type.setdefault(sp, {}).setdefault(ft, [])
            if fn not in by_split_and_type[sp][ft]:
                by_split_and_type[sp][ft].append(fn)

        holdout_files = set()
        for ft, flist in by_split_and_type.get("holdout", {}).items():
            holdout_files.update(flist)

        train_candidates = []
        for ft in sorted(by_split_and_type.get("train", {}).keys()):
            for fn in sorted(by_split_and_type["train"][ft]):
                train_candidates.append(fn)

        rng = random.Random(self.seed)
        # Stratified validation selection across formats from the train candidates
        val_files: Set[str] = set()
        for ft in sorted(by_split_and_type.get("train", {}).keys()):
            flist = sorted(by_split_and_type["train"][ft])
            if len(flist) >= 4:
                # Select 1 validation file per format that has ample samples
                pick = rng.choice(flist)
                val_files.add(pick)

        # Ensure total val files reaches target ratio (approx 5 files out of 37)
        target_val = max(1, int(round(len(train_candidates) * self.val_file_ratio)))
        remaining_candidates = [f for f in train_candidates if f not in val_files]
        while len(val_files) < target_val and remaining_candidates:
            pick = rng.choice(remaining_candidates)
            val_files.add(pick)
            remaining_candidates.remove(pick)

        train_files = set(train_candidates) - val_files

        if not verify_partition_disjointness(train_files, val_files, holdout_files):
            raise RuntimeError("CRITICAL: Partition file disjointness verification failed!")

        return train_files, val_files, holdout_files

    def _assign_metadata(
        self,
        metas_raw: List[Dict[str, str]],
        train_files: Set[str],
        val_files: Set[str],
        holdout_files: Set[str],
    ) -> List[Dict[str, object]]:
        """Assign forensic metadata based on context_mode without leakage."""
        metas: List[Dict[str, object]] = []

        if self.context_mode == "metadata_independent":
            for r in metas_raw:
                fn = r["source_file"]
                part = "train" if fn in train_files else ("val" if fn in val_files else "holdout")
                metas.append({
                    "fragment_id": r["fragment_id"],
                    "source_file": fn,
                    "file_type": r["file_type"],
                    "original_position": int(r["original_position"]),
                    "fragment_size": int(r["fragment_size"]),
                    "split": part,
                    "similar_group": r["similar_group"],
                    "variant": "orig",
                    "corruption_level": 0.0,
                    "allocation_status": "unallocated",
                    "physical_offset": 0,
                    "cluster": 0,
                    "filesystem_context": {"fs_type": "RAW", "volume_id": "CARVED"},
                })

        elif self.context_mode == "simulated_layout":
            # Simulate non-leaking filesystem layout with multi-fragment runs and interleaving
            rng = random.Random(self.seed + 101)
            file_clusters: Dict[str, List[int]] = {}
            current_cluster = 100
            for r in metas_raw:
                fn = r["source_file"]
                if fn not in file_clusters:
                    file_clusters[fn] = []
                # Interleaved simulated run
                current_cluster += rng.randint(1, 4)
                file_clusters[fn].append(current_cluster)

            file_frag_counter: Dict[str, int] = {}
            for r in metas_raw:
                fn = r["source_file"]
                idx = file_frag_counter.get(fn, 0)
                file_frag_counter[fn] = idx + 1
                sim_cluster = file_clusters[fn][idx]
                part = "train" if fn in train_files else ("val" if fn in val_files else "holdout")
                metas.append({
                    "fragment_id": r["fragment_id"],
                    "source_file": fn,
                    "file_type": r["file_type"],
                    "original_position": int(r["original_position"]),
                    "fragment_size": int(r["fragment_size"]),
                    "split": part,
                    "similar_group": r["similar_group"],
                    "variant": "orig",
                    "corruption_level": 0.0,
                    "allocation_status": "allocated",
                    "physical_offset": sim_cluster * FRAGMENT_SIZE,
                    "cluster": sim_cluster,
                    "filesystem_context": {"fs_type": "FSIM_SIM", "volume_id": "DISK0"},
                })

        elif self.context_mode == "ground_truth_debug":
            # Preserved for ablation comparison with v1 (leaks offset)
            for r in metas_raw:
                fn = r["source_file"]
                part = "train" if fn in train_files else ("val" if fn in val_files else "holdout")
                metas.append({
                    "fragment_id": r["fragment_id"],
                    "source_file": fn,
                    "file_type": r["file_type"],
                    "original_position": int(r["original_position"]),
                    "fragment_size": int(r["fragment_size"]),
                    "split": part,
                    "similar_group": r["similar_group"],
                    "variant": "orig",
                    "corruption_level": 0.0,
                    "allocation_status": "allocated",
                    "physical_offset": int(r["offset_in_source"]),
                    "cluster": int(r["original_position"]),
                    "filesystem_context": {"fs_type": "FSIM", "volume_id": "CORPUS"},
                })

        return metas

    def build(self) -> Dict[str, object]:
        """Execute leakage-free dataset building."""
        set_seeds(self.seed)
        with timer("pair dataset build v2", logger):
            metas_raw, blobs = self.load_corpus()
            block = FRAGMENT_SIZE
            base = [np.frombuffer(_pad(b, block), dtype=np.uint8) for b in blobs]

            train_files, val_files, holdout_files = self._partition_files(metas_raw)
            metas = self._assign_metadata(metas_raw, train_files, val_files, holdout_files)

            # Corruption augmentation applied strictly to training partition fragments
            data: List[np.ndarray] = list(base)
            variants: Dict[int, List[int]] = {i: [i] for i in range(len(base))}

            for i, m in enumerate(metas_raw):
                if m["source_file"] not in train_files:
                    continue
                if self.rng.random() < self.corruption_aug_frac:
                    lvl = self.rng.choice(CORRUPTION_LEVELS)
                    arr = base[i].copy()
                    n_corrupt = int(round(arr.size * lvl))
                    if n_corrupt > 0:
                        idx = self.rng.sample(range(arr.size), n_corrupt)
                        noise = np.frombuffer(self.rng.randbytes(n_corrupt), dtype=np.uint8)
                        arr[idx] = noise
                    data.append(arr)
                    metas.append({**metas[i], "variant": f"corr{lvl:.2f}", "corruption_level": lvl})
                    variants[i].append(len(data) - 1)

            data_matrix = np.stack(data)
            engine = FeatureEngineV2(data_matrix, metas)
            logger.info("feature engine v2: %d fragments x %d single features",
                        engine.n_, engine.single_matrix_.shape[1])

            train_rows = [i for i, m in enumerate(metas) if m["variant"] == "orig" and m["split"] == "train"]
            val_rows = [i for i, m in enumerate(metas) if m["variant"] == "orig" and m["split"] == "val"]
            hold_rows = [i for i, m in enumerate(metas) if m["variant"] == "orig" and m["split"] == "holdout"]

            def make_dataset(rows: List[int], partition_name: str, seed_offset: int) -> PairDatasetV2:
                pair_idx, y_same, y_next, hard, grp_a, grp_b = generate_pairs_v2(
                    metas,
                    rows,
                    neg_ratio=self.neg_ratio,
                    same_ratio=self.same_ratio,
                    hard_negative_frac=HARD_NEGATIVE_FRACTION,
                    seed=self.seed + seed_offset,
                    partition_name=partition_name,
                )
                r_idx = np.empty_like(pair_idx)
                for k, (li, lj) in enumerate(pair_idx):
                    oi = rows[li]
                    oj = rows[lj]
                    # For training, sample corruption variants; for val/holdout keep clean
                    vi = self.rng.choice(variants[oi]) if partition_name == "train" else oi
                    vj = self.rng.choice(variants[oj]) if partition_name == "train" else oj
                    r_idx[k] = (vi, vj)

                X = engine.pair_matrix_vec(r_idx)
                return PairDatasetV2(
                    pair_idx=r_idx,
                    X=X,
                    y_same=y_same,
                    y_next=y_next,
                    hard_negative=hard,
                    group_a=grp_a,
                    group_b=grp_b,
                    feature_names=[n for n, _ in PAIR_NAMES],
                    partition=partition_name,
                )

            raw_train_ds = make_dataset(train_rows, "train", 0)
            raw_val_ds = make_dataset(val_rows, "val", 100)
            raw_hold_ds = make_dataset(hold_rows, "holdout", 200)

            # Fit normalizer strictly on raw_train_ds.X
            normalizer = FeatureNormalizerV2(method=self.normalize_method)
            normalizer.fit(raw_train_ds.X)
            engine.normalizer = normalizer

            train_ds = PairDatasetV2(
                pair_idx=raw_train_ds.pair_idx,
                X=normalizer.transform(raw_train_ds.X),
                y_same=raw_train_ds.y_same,
                y_next=raw_train_ds.y_next,
                hard_negative=raw_train_ds.hard_negative,
                group_a=raw_train_ds.group_a,
                group_b=raw_train_ds.group_b,
                feature_names=raw_train_ds.feature_names,
                partition="train",
            )
            val_ds = PairDatasetV2(
                pair_idx=raw_val_ds.pair_idx,
                X=normalizer.transform(raw_val_ds.X),
                y_same=raw_val_ds.y_same,
                y_next=raw_val_ds.y_next,
                hard_negative=raw_val_ds.hard_negative,
                group_a=raw_val_ds.group_a,
                group_b=raw_val_ds.group_b,
                feature_names=raw_val_ds.feature_names,
                partition="val",
            )
            holdout_ds = PairDatasetV2(
                pair_idx=raw_hold_ds.pair_idx,
                X=normalizer.transform(raw_hold_ds.X),
                y_same=raw_hold_ds.y_same,
                y_next=raw_hold_ds.y_next,
                hard_negative=raw_hold_ds.hard_negative,
                group_a=raw_hold_ds.group_a,
                group_b=raw_hold_ds.group_b,
                feature_names=raw_hold_ds.feature_names,
                partition="holdout",
            )

            return {
                "engine": engine,
                "metas": metas,
                "variants": variants,
                "train_ds": train_ds,
                "val_ds": val_ds,
                "holdout_ds": holdout_ds,
                "raw_train_ds": raw_train_ds,
                "raw_val_ds": raw_val_ds,
                "raw_hold_ds": raw_hold_ds,
                "normalizer": normalizer,
                "train_files": sorted(train_files),
                "val_files": sorted(val_files),
                "holdout_files": sorted(holdout_files),
                "context_mode": self.context_mode,
            }
