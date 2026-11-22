"""Unit and integration tests for ForenSight v2 feature engineering and normalizer."""
import pytest
import numpy as np

from features_v2 import FeatureEngineV2, PAIR_NAMES, FAMILIES, ABLATION_MASKS
from preprocessing_v2.normalize_v2 import FeatureNormalizerV2


def test_feature_engine_v2_shapes_and_numeric_hygiene():
    """Verify single and pair feature matrix dimensions and lack of NaNs/Infs."""
    # Synthetic 5-fragment corpus
    n = 5
    block_size = 512
    rng = np.random.RandomState(42)
    data = rng.randint(0, 256, size=(n, block_size), dtype=np.uint8)
    metas = [
        {"physical_offset": 0, "cluster": 0, "allocation_status": "unallocated", "source_file": f"f{i}"}
        for i in range(n)
    ]

    engine = FeatureEngineV2(data, metas)

    # Contract: 400 single features, 362 pair features
    assert engine.single_matrix_.shape == (n, 400)
    assert not np.isnan(engine.single_matrix_).any()
    assert not np.isinf(engine.single_matrix_).any()

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    pair_mat_loop = engine.pair_matrix(pairs)
    pair_mat_vec = engine.pair_matrix_vec(np.array(pairs))

    assert pair_mat_loop.shape == (len(pairs), 362)
    assert pair_mat_vec.shape == (len(pairs), 362)
    assert not np.isnan(pair_mat_vec).any()
    assert not np.isinf(pair_mat_vec).any()

    # Vectorized computation matches looped computation within numeric precision
    max_diff = np.max(np.abs(pair_mat_loop - pair_mat_vec))
    assert max_diff < 1e-4, f"Vectorized pair calculation diverged from loop by {max_diff}"


def test_feature_normalizer_v2_strict_isolation(tmp_path):
    """Verify that FeatureNormalizerV2 isolates training statistics from validation data."""
    rng = np.random.RandomState(42)
    X_train = rng.normal(loc=10.0, scale=5.0, size=(100, 20)).astype(np.float32)
    X_val = rng.normal(loc=50.0, scale=20.0, size=(50, 20)).astype(np.float32)

    norm = FeatureNormalizerV2(method="standard")
    norm.fit(X_train)

    assert norm.fitted_ is True
    assert norm.mean_ is not None and norm.std_ is not None
    orig_mean = norm.mean_.copy()
    orig_std = norm.std_.copy()

    # Standardize train
    X_train_trans = norm.transform(X_train)
    assert np.allclose(X_train_trans.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(X_train_trans.std(axis=0), 1.0, atol=1e-3)

    # Standardize out-of-sample val
    X_val_trans = norm.transform(X_val)

    # Verify fitted parameters did NOT shift when transforming X_val (no leak)
    assert np.array_equal(norm.mean_, orig_mean)
    assert np.array_equal(norm.std_, orig_std)

    # Verify save and load
    save_file = tmp_path / "normalizer.joblib"
    norm.save(save_file)
    loaded = FeatureNormalizerV2.load(save_file)

    assert loaded.fitted_ is True
    assert np.array_equal(loaded.mean_, orig_mean)
    assert np.array_equal(loaded.std_, orig_std)
    loaded_trans = loaded.transform(X_val)
    assert np.array_equal(loaded_trans, X_val_trans)


def test_feature_normalizer_v2_minmax_and_robust():
    """Verify minmax and robust scaling modes."""
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)

    norm_minmax = FeatureNormalizerV2(method="minmax")
    X_mm = norm_minmax.fit_transform(X)
    assert np.allclose(X_mm[0], [0.0, 0.0])
    assert np.allclose(X_mm[-1], [1.0, 1.0])

    norm_robust = FeatureNormalizerV2(method="robust")
    X_rb = norm_robust.fit_transform(X)
    assert X_rb.shape == X.shape
