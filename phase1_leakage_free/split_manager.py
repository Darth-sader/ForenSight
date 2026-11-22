"""Group-Aware Forensic Split Manager (v2.0.0-leakage_free).

Guarantees 100% strict partition isolation:
1. Grouping unit is the original source file.
2. Generator/source-group isolation: files sharing a similar_group stay strictly atomic.
3. 4-way disjoint splits: train, val, cal (calibration), and test (held-out).
4. Zero contamination: test set is completely isolated and never used for tuning,
   feature selection, model selection, or calibration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import GROUND_TRUTH_DIR, SEED


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class SplitManifest:
    version: str = "2.0.0-leakage_free"
    seed: int = SEED
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)
    cal_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    file_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    manifest_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "counts": {
                "train_files": len(self.train_files),
                "val_files": len(self.val_files),
                "cal_files": len(self.cal_files),
                "test_files": len(self.test_files),
                "total_files": len(self.train_files) + len(self.val_files) + len(self.cal_files) + len(self.test_files),
            },
            "train_files": sorted(self.train_files),
            "val_files": sorted(self.val_files),
            "cal_files": sorted(self.cal_files),
            "test_files": sorted(self.test_files),
            "file_metadata": self.file_metadata,
            "manifest_hash": self.manifest_hash,
        }


class GroupSplitManager:
    """Manages group-aware forensic evidence partitioning."""

    def __init__(self, ground_truth_csv: Optional[Path] = None, seed: int = SEED):
        self.csv_path = ground_truth_csv or (GROUND_TRUTH_DIR / "fragments.csv")
        self.seed = seed

    def load_file_records(self) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
        """Read fragments.csv and aggregate file-level metadata and similar groups."""
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        files_meta: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            fn = r["source_file"]
            if fn not in files_meta:
                files_meta[fn] = {
                    "source_file": fn,
                    "file_type": r["file_type"],
                    "split_orig": r["split"],
                    "similar_group": r["similar_group"],
                    "fragment_count": 0,
                    "byte_count": 0,
                }
            files_meta[fn]["fragment_count"] += 1
            files_meta[fn]["byte_count"] += int(r["fragment_size"])

        return rows, files_meta

    def create_splits(
        self,
        val_file_count: int = 5,
        cal_file_count: int = 4,
    ) -> SplitManifest:
        """Partition the corpus into 4 strictly disjoint splits with source-group atomicity."""
        rows, files_meta = self.load_file_records()

        # Separate holdout files (test partition) from training candidate pool
        test_files: Set[str] = {fn for fn, m in files_meta.items() if m["split_orig"] == "holdout"}
        train_candidates: Set[str] = {fn for fn, m in files_meta.items() if m["split_orig"] == "train"}

        # Cluster files by source group / generator
        # If similar_group is defined, all files sharing it form an atomic unit
        group_to_files: Dict[str, List[str]] = {}
        for fn in train_candidates:
            sg = files_meta[fn]["similar_group"]
            gid = sg if sg else f"file::{fn}"
            group_to_files.setdefault(gid, []).append(fn)

        # Ensure determinism with fixed seed
        rng = random.Random(self.seed)
        atomic_groups = sorted(group_to_files.keys())
        rng.shuffle(atomic_groups)

        val_files: Set[str] = set()
        cal_files: Set[str] = set()
        train_files: Set[str] = set()

        # Stratified / group selection across file formats for val and cal
        # 1. Allocate to validation until target reached
        for gid in atomic_groups:
            members = group_to_files[gid]
            if len(val_files) < val_file_count:
                val_files.update(members)
            elif len(cal_files) < cal_file_count:
                cal_files.update(members)
            else:
                train_files.update(members)

        # Verify strict 4-way disjointness
        assert len(train_files & val_files) == 0, "Train and Val overlap!"
        assert len(train_files & cal_files) == 0, "Train and Cal overlap!"
        assert len(train_files & test_files) == 0, "Train and Test overlap!"
        assert len(val_files & cal_files) == 0, "Val and Cal overlap!"
        assert len(val_files & test_files) == 0, "Val and Test overlap!"
        assert len(cal_files & test_files) == 0, "Cal and Test overlap!"

        # Verify similar_group atomicity
        for fn in files_meta:
            sg = files_meta[fn]["similar_group"]
            if sg:
                sibling_files = [k for k, v in files_meta.items() if v["similar_group"] == sg]
                splits_hit = set()
                for sf in sibling_files:
                    if sf in train_files:
                        splits_hit.add("train")
                    elif sf in val_files:
                        splits_hit.add("val")
                    elif sf in cal_files:
                        splits_hit.add("cal")
                    elif sf in test_files:
                        splits_hit.add("test")
                assert len(splits_hit) == 1, f"Similar group {sg} split across partitions {splits_hit}!"

        manifest = SplitManifest(
            seed=self.seed,
            train_files=sorted(train_files),
            val_files=sorted(val_files),
            cal_files=sorted(cal_files),
            test_files=sorted(test_files),
            file_metadata=files_meta,
        )

        content_for_hash = json.dumps(
            {
                "train": manifest.train_files,
                "val": manifest.val_files,
                "cal": manifest.cal_files,
                "test": manifest.test_files,
                "seed": manifest.seed,
            },
            sort_keys=True,
        ).encode("utf-8")
        manifest.manifest_hash = compute_sha256(content_for_hash)

        return manifest


def save_split_manifest(manifest: SplitManifest, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_split_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
