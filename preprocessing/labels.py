"""Label generation and corpus pair-dataset assembly (doc §8, §11, §26).

Loads the controlled corpus, augments fragments with controlled corruption
variants (scenario D robustness, doc §7-D), computes features once with the
FeatureEngine and emits labelled pair datasets:

* train/val pairs from training files (GroupShuffleSplit by source file)
* holdout pairs from completely unseen files (generalisation test, doc §27)

The same original file never contributes fragments to both train and test.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from common import logger, set_seeds, timer
from config import (CORRUPTION_LEVELS, FRAGMENT_SIZE, FRAGMENTS_DIR,
                    GROUND_TRUTH_DIR, HARD_NEGATIVE_FRACTION, PAIR_NEGATIVE_RATIO,
                    SAME_PAIR_RATIO, SEED)
from features import FeatureEngine, PAIR_NAMES
from preprocessing.sampling import PairDataset, generate_pairs


def _pad(buf: bytes, size: int) -> bytes:
    return buf if len(buf) >= size else buf.ljust(size, b"\x00")


class CorpusPairBuilder:
    def __init__(self, seed: int = SEED,
                 corruption_aug_frac: float = 0.35,
                 neg_ratio: float = PAIR_NEGATIVE_RATIO,
                 same_ratio: float = SAME_PAIR_RATIO):
        self.seed = seed
        self.rng = random.Random(seed)
        self.corruption_aug_frac = corruption_aug_frac
        self.neg_ratio = neg_ratio
        self.same_ratio = same_ratio

    # -------------------------------------------------------------- loading
    def load_corpus(self) -> Tuple[List[Dict[str, object]], List[bytes]]:
        rows = []
        with open(GROUND_TRUTH_DIR / "fragments.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        rows.sort(key=lambda r: int(r["fragment_id"][1:]))
        blobs = [(FRAGMENTS_DIR / "blobs" / f"{r['fragment_id']}.bin").read_bytes() for r in rows]
        return rows, blobs

    # -------------------------------------------------------------- build
    def build(self) -> Dict[str, object]:
        set_seeds(self.seed)
        with timer("pair dataset build", logger):
            metas_raw, blobs = self.load_corpus()
            block = FRAGMENT_SIZE
            base = [np.frombuffer(_pad(b, block), dtype=np.uint8) for b in blobs]

            # corruption augmentation of training fragments (scenario D robustness)
            data: List[np.ndarray] = list(base)
            variants: Dict[int, List[int]] = {i: [i] for i in range(len(base))}
            metas: List[Dict[str, object]] = []
            for r in metas_raw:
                metas.append({
                    "fragment_id": r["fragment_id"], "source_file": r["source_file"],
                    "file_type": r["file_type"], "original_position": int(r["original_position"]),
                    "fragment_size": int(r["fragment_size"]), "split": r["split"],
                    "similar_group": r["similar_group"], "variant": "orig", "corruption_level": 0.0,
                    "allocation_status": "allocated", "physical_offset": int(r["offset_in_source"]),
                    "cluster": int(r["original_position"]),
                    "filesystem_context": {"fs_type": "FSIM", "volume_id": "CORPUS"},
                })
            for i, r in enumerate(metas_raw):
                if r["split"] != "train":
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
            engine = FeatureEngine(data_matrix, metas)
            logger.info("feature engine: %d fragments x %d features", engine.n_,
                        engine.single_matrix_.shape[1])

            # row indices restricted to original (non-augmented) fragments
            train_rows = [i for i, m in enumerate(metas) if m["variant"] == "orig"
                          and m["split"] == "train"]
            hold_rows = [i for i, m in enumerate(metas) if m["variant"] == "orig"
                         and m["split"] == "holdout"]

            def make_dataset(rows: List[int]) -> PairDataset:
                p_idx, y_same, y_next, hard, group = generate_pairs(
                    metas, rows, neg_ratio=self.neg_ratio, same_ratio=self.same_ratio,
                    hard_negative_frac=HARD_NEGATIVE_FRACTION, seed=self.seed)
                # sample corrupted variants for both sides of each pair
                r_idx = np.empty_like(p_idx)
                for k, (li, lj) in enumerate(p_idx):
                    oi = rows[li]
                    oj = rows[lj]
                    vi = self.rng.choice(variants[oi])
                    vj = self.rng.choice(variants[oj])
                    r_idx[k] = (vi, vj)
                X = engine.pair_matrix_vec(r_idx)
                return PairDataset(pair_idx=r_idx, X=X, y_same=y_same, y_next=y_next,
                                   hard_negative=hard, group=group,
                                   feature_names=[n for n, _ in PAIR_NAMES])

            train_ds = make_dataset(train_rows)
            hold_ds = make_dataset(hold_rows)

            # file-level group split of the training pairs into train/val (doc §26)
            unique_groups = np.unique(train_ds.group)
            rng = np.random.RandomState(self.seed)
            perm = rng.permutation(len(unique_groups))
            n_val = max(1, int(0.15 * len(unique_groups)))
            val_groups = set(unique_groups[perm[:n_val]].tolist())
            is_val = np.isin(train_ds.group, list(val_groups))

            return {
                "engine": engine,
                "metas": metas,
                "variants": variants,
                "train_ds": train_ds[~is_val],
                "val_ds": train_ds[is_val],
                "holdout_ds": hold_ds,
                "n_train_groups": int(len(unique_groups) - n_val),
                "n_val_groups": int(n_val),
            }
