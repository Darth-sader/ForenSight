"""Automated Leakage Detection Test Suite (v2.0.0-leakage_free).

Enforces strict mathematical and procedural checks to guarantee that zero
ground truth, cross-split pairs, or preprocessing contamination enter the
ForenSight inference pipeline.
"""
import copy
import pytest
import numpy as np

from phase1_leakage_free.fragment_schema_v2 import (
    ObservableFragment,
    GroundTruthFragment,
    validate_observable_record,
    SchemaLeakageError,
    GROUND_TRUTH_ONLY_FIELDS,
)
from phase1_leakage_free.split_manager import GroupSplitManager
from phase1_leakage_free.pair_generator_v2 import PairGeneratorV2
from phase1_leakage_free.leakage_free_features import LeakageFreeFeatureEngine
from phase1_leakage_free.preprocessing_v2 import LeakageFreePreprocessor


def test_ground_truth_fields_rejected_by_validator():
    """Verify that any record containing ground-truth fields fails loudly."""
    # Clean observable dict
    clean_record = {
        "fragment_id": "f0001",
        "observed_offset": 51200,
        "fragment_bytes": b"\x00" * 512,
        "fragment_length": 512,
        "allocation_status": "unallocated",
        "filesystem_metadata": {"fs_type": "RAW"},
        "filesystem_region": "data_area",
        "acquisition_source": "disk_image",
    }
    # Should pass without error
    validate_observable_record(clean_record)

    # Inject every ground-truth field one-by-one and verify failure
    for gt_field in GROUND_TRUTH_ONLY_FIELDS:
        leaky_record = copy.deepcopy(clean_record)
        leaky_record[gt_field] = 42 if "offset" in gt_field or "index" in gt_field else "leak"
        with pytest.raises(SchemaLeakageError, match="CRITICAL SCHEMA LEAKAGE DETECTED"):
            validate_observable_record(leaky_record)


def test_shared_original_files_across_splits():
    """Verify that train, val, cal, and test sets share 0 source files and respect generator atomicity."""
    mgr = GroupSplitManager()
    manifest = mgr.create_splits(val_file_count=5, cal_file_count=4)

    train_set = set(manifest.train_files)
    val_set = set(manifest.val_files)
    cal_set = set(manifest.cal_files)
    test_set = set(manifest.test_files)

    # 4-way disjointness checks
    assert len(train_set & val_set) == 0, f"Leakage: Train/Val overlap on {train_set & val_set}"
    assert len(train_set & cal_set) == 0, f"Leakage: Train/Cal overlap on {train_set & cal_set}"
    assert len(train_set & test_set) == 0, f"Leakage: Train/Test overlap on {train_set & test_set}"
    assert len(val_set & cal_set) == 0, f"Leakage: Val/Cal overlap on {val_set & cal_set}"
    assert len(val_set & test_set) == 0, f"Leakage: Val/Test overlap on {val_set & test_set}"
    assert len(cal_set & test_set) == 0, f"Leakage: Cal/Test overlap on {cal_set & test_set}"

    # Verify similar_group generator atomicity
    files_meta = manifest.file_metadata
    for fn, meta in files_meta.items():
        sg = meta["similar_group"]
        if sg:
            siblings = [k for k, v in files_meta.items() if v["similar_group"] == sg]
            assigned_splits = set()
            for s in siblings:
                if s in train_set:
                    assigned_splits.add("train")
                elif s in val_set:
                    assigned_splits.add("val")
                elif s in cal_set:
                    assigned_splits.add("cal")
                elif s in test_set:
                    assigned_splits.add("test")
            assert len(assigned_splits) == 1, f"Generator leakage: similar group {sg} split across {assigned_splits}"


def test_cross_split_fragment_pairs_rejected():
    """Verify that pair generation samples exclusively within permitted partition boundaries."""
    # Create synthetic fragments
    frags: list[GroundTruthFragment] = []
    for file_id in ["train_1.jpg", "train_2.jpg", "test_1.jpg"]:
        for pos in range(3):
            obs = ObservableFragment(
                fragment_id=f"{file_id}_{pos}",
                observed_offset=0,
                fragment_bytes=b"\xaa" * 512,
                fragment_length=512,
                allocation_status="unallocated",
            )
            frags.append(GroundTruthFragment(
                observable=obs,
                source_file_id=file_id,
                logical_offset=pos * 512,
                logical_sequence_index=pos,
                original_fragment_id=f"{file_id}_{pos}",
                original_bytes=b"\xaa" * 512,
                split="train" if "train" in file_id else "test",
            ))

    engine = LeakageFreeFeatureEngine([f.to_observable() for f in frags])
    gen = PairGeneratorV2(seed=42)

    train_files = {"train_1.jpg", "train_2.jpg"}
    train_ds = gen.generate_pairs_for_partition(
        fragments=frags,
        allowed_files=train_files,
        engine=engine,
        partition_name="train",
    )

    # Assert 0 pairs involve test_1.jpg
    for prov in train_ds.provenance:
        assert prov.source_file_a in train_files, f"Cross-split leakage: {prov.source_file_a} in train pair"
        assert prov.source_file_b in train_files, f"Cross-split leakage: {prov.source_file_b} in train pair"
        assert prov.source_file_a != "test_1.jpg"
        assert prov.source_file_b != "test_1.jpg"


def test_test_set_preprocessing_isolation():
    """Verify that preprocessor statistics are fitted strictly on train and unmodified by test sets."""
    rng = np.random.RandomState(42)
    X_train = rng.normal(loc=5.0, scale=2.0, size=(100, 10)).astype(np.float32)
    X_test = rng.normal(loc=100.0, scale=50.0, size=(50, 10)).astype(np.float32)

    prep = LeakageFreePreprocessor(scaler_type="standard")
    prep.fit(X_train)

    assert prep.scaler_ is not None
    orig_mean = prep.scaler_.mean_.copy()
    orig_var = prep.scaler_.var_.copy()

    # Apply to test set
    _ = prep.transform(X_test)

    # Assert train statistics did not shift
    assert np.array_equal(prep.scaler_.mean_, orig_mean), "Leakage: scaler mean altered by test set transform!"
    assert np.array_equal(prep.scaler_.var_, orig_var), "Leakage: scaler var altered by test set transform!"


def test_calibration_test_isolation():
    """Verify that calibration set files do not overlap with test set files."""
    mgr = GroupSplitManager()
    manifest = mgr.create_splits(val_file_count=5, cal_file_count=4)

    cal_files = set(manifest.cal_files)
    test_files = set(manifest.test_files)
    assert len(cal_files & test_files) == 0, f"Calibration/Test overlap detected: {cal_files & test_files}"
    assert len(cal_files) > 0, "Calibration set must not be empty"
    assert len(test_files) > 0, "Test set must not be empty"


def test_no_logical_offset_features_in_inference():
    """Verify that in metadata-independent mode, physical_adjacent is NOT 1.0 for true next fragments."""
    # Consecutive fragments with 0 observed offset
    obs_a = ObservableFragment(
        fragment_id="fa",
        observed_offset=0,
        fragment_bytes=b"\x01" * 512,
        fragment_length=512,
        allocation_status="unallocated",
    )
    obs_b = ObservableFragment(
        fragment_id="fb",
        observed_offset=0,
        fragment_bytes=b"\x02" * 512,
        fragment_length=512,
        allocation_status="unallocated",
    )

    engine = LeakageFreeFeatureEngine([obs_a, obs_b], context_mode="metadata_independent")
    row = engine.pair_row(0, 1)

    from features import PAIR_NAMES
    adj_idx = [name for name, _ in PAIR_NAMES].index("physical_adjacent")
    assert row[adj_idx] == 0.0, f"Logical offset leakage detected! physical_adjacent is {row[adj_idx]}, expected 0.0"
