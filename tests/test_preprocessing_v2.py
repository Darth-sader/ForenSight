"""Unit and integration tests for ForenSight v2 preprocessing components."""
import pytest
import numpy as np

from preprocessing_v2.sampling_v2 import (
    PairDatasetV2,
    generate_pairs_v2,
    verify_partition_disjointness,
)
from preprocessing_v2.normalize_v2 import FeatureNormalizerV2
from preprocessing_v2.labels_v2 import CorpusPairBuilderV2


def test_verify_partition_disjointness():
    """Verify that partition disjointness validator strictly detects overlaps."""
    train_files = {"file1.jpg", "file2.jpg", "file3.png"}
    val_files = {"file4.pdf", "file5.docx"}
    holdout_files = {"file6.zip", "file7.xlsx"}

    assert verify_partition_disjointness(train_files, val_files, holdout_files) is True

    # Introduce leak between train and val
    leaky_val = {"file2.jpg", "file5.docx"}
    assert verify_partition_disjointness(train_files, leaky_val, holdout_files) is False

    # Introduce leak between train and holdout
    leaky_holdout = {"file1.jpg", "file7.xlsx"}
    assert verify_partition_disjointness(train_files, val_files, leaky_holdout) is False


def test_generate_pairs_v2_strict_partition_isolation():
    """Verify that generate_pairs_v2 samples exclusively within the provided partition slice."""
    # Synthetic corpus with 4 files across 2 types
    metas = [
        # File 1 (JPEG, 3 fragments)
        {"source_file": "f1.jpg", "file_type": "JPEG", "original_position": 0},
        {"source_file": "f1.jpg", "file_type": "JPEG", "original_position": 1},
        {"source_file": "f1.jpg", "file_type": "JPEG", "original_position": 2},
        # File 2 (JPEG, 3 fragments)
        {"source_file": "f2.jpg", "file_type": "JPEG", "original_position": 0},
        {"source_file": "f2.jpg", "file_type": "JPEG", "original_position": 1},
        {"source_file": "f2.jpg", "file_type": "JPEG", "original_position": 2},
        # File 3 (PNG, 3 fragments) - belongs to val
        {"source_file": "f3.png", "file_type": "PNG", "original_position": 0},
        {"source_file": "f3.png", "file_type": "PNG", "original_position": 1},
        {"source_file": "f3.png", "file_type": "PNG", "original_position": 2},
        # File 4 (PNG, 3 fragments) - belongs to holdout
        {"source_file": "f4.png", "file_type": "PNG", "original_position": 0},
        {"source_file": "f4.png", "file_type": "PNG", "original_position": 1},
        {"source_file": "f4.png", "file_type": "PNG", "original_position": 2},
    ]

    # Only pass File 1 and File 2 rows as training partition (rows 0-5)
    train_rows = list(range(6))
    pair_idx, y_same, y_next, hard, grp_a, grp_b = generate_pairs_v2(
        metas=metas,
        fragment_rows=train_rows,
        neg_ratio=2.0,
        same_ratio=1.0,
        hard_negative_frac=0.5,
        seed=42,
        partition_name="train",
    )

    # Every sampled local index MUST be < len(train_rows)
    assert np.all(pair_idx[:, 0] < len(train_rows))
    assert np.all(pair_idx[:, 1] < len(train_rows))
    assert np.all(pair_idx >= 0)

    # Map back to global rows and verify zero fragments from f3 or f4 leaked
    global_a = [train_rows[li] for li in pair_idx[:, 0]]
    global_b = [train_rows[lj] for lj in pair_idx[:, 1]]
    files_a = {metas[ga]["source_file"] for ga in global_a}
    files_b = {metas[gb]["source_file"] for gb in global_b}

    assert files_a.issubset({"f1.jpg", "f2.jpg"})
    assert files_b.issubset({"f1.jpg", "f2.jpg"})
    assert "f3.png" not in files_a and "f3.png" not in files_b
    assert "f4.png" not in files_a and "f4.png" not in files_b


def test_pair_dataset_v2_slicing():
    """Test PairDatasetV2 indexing and slicing behavior."""
    n = 10
    ds = PairDatasetV2(
        pair_idx=np.zeros((n, 2), dtype=np.int64),
        X=np.ones((n, 5), dtype=np.float32),
        y_same=np.ones(n, dtype=np.int32),
        y_next=np.zeros(n, dtype=np.int32),
        hard_negative=np.zeros(n, dtype=bool),
        group_a=np.arange(n, dtype=np.int64),
        group_b=np.arange(n, dtype=np.int64),
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        partition="test",
    )

    assert len(ds) == n
    sub = ds[np.array([True, False] * 5)]
    assert len(sub) == 5
    assert sub.X.shape == (5, 5)
    assert sub.partition == "test"


def test_corpus_pair_builder_v2_metadata_independent_no_leakage():
    """Verify that CorpusPairBuilderV2 in metadata_independent mode eliminates LR-01 leakage."""
    builder = CorpusPairBuilderV2(
        seed=42,
        context_mode="metadata_independent",
        corruption_aug_frac=0.1,
        neg_ratio=1.0,
        same_ratio=0.5,
    )
    bundle = builder.build()

    train_files = set(bundle["train_files"])
    val_files = set(bundle["val_files"])
    holdout_files = set(bundle["holdout_files"])

    # Disjointness check
    assert verify_partition_disjointness(train_files, val_files, holdout_files) is True
    assert len(train_files) > 0
    assert len(val_files) > 0
    assert len(holdout_files) > 0

    # Verify metadata-independent properties
    metas = bundle["metas"]
    for m in metas:
        if m["variant"] == "orig":
            assert m["physical_offset"] == 0
            assert m["cluster"] == 0
            assert m["allocation_status"] == "unallocated"

    # In raw_train_ds, physical_adjacent feature should NOT be 1.0 for positive pairs
    engine = bundle["engine"]
    raw_train_ds = bundle["raw_train_ds"]
    pair_names = raw_train_ds.feature_names
    phys_adj_idx = pair_names.index("physical_adjacent")

    pos_mask = raw_train_ds.y_next == 1
    phys_adj_vals = raw_train_ds.X[pos_mask, phys_adj_idx]
    # In v1, this was 100% 1.0 (leakage). In v2 metadata_independent, it MUST be 0.0!
    assert np.all(phys_adj_vals == 0.0), "Leakage LR-01 detected: physical_adjacent is non-zero in metadata_independent mode!"
