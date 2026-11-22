"""Baseline Re-Evaluation Training Pipeline (v2.0.0-leakage_free).

Trains and independently evaluates all 6 existing model architectures on:
1. Same-file prediction (y_same)
2. Next-fragment prediction (y_next)

Under strictly leakage-free conditions:
- Zero ground-truth offset or cluster leakage.
- Zero cross-split fragment pairs.
- Preprocessing strictly fitted on training data only.
- Strict 4-way disjoint group-aware partitioning (train, val, cal, test).
- Complete metric reporting: ROC-AUC, PR-AUC, Precision, Recall, F1, Brier, Log Loss, ECE, Precision@K, Recall@K.
- All artifacts isolated in phase1_leakage_free/artifacts/ (zero overwriting of v1 models).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import logger, set_seeds, timer
from config import (
    CORPUS_PLAN,
    FRAGMENT_SIZE,
    FRAGMENTS_DIR,
    GROUND_TRUTH_DIR,
    SEED,
)
from models.cnn_encoder import SiameseCNN
from models.logistic import LogisticModel
from models.mlp import MLPTorchModel
from models.random_forest import RandomForestModel
from models.siamese import SiameseNetwork
from models.xgboost import XGBoostModel
from phase1_leakage_free.fragment_schema_v2 import (
    GroundTruthFragment,
    ObservableFragment,
    validate_observable_record,
)
from phase1_leakage_free.leakage_free_features import LeakageFreeFeatureEngine
from phase1_leakage_free.pair_generator_v2 import PairDatasetV2, PairGeneratorV2
from phase1_leakage_free.preprocessing_v2 import LeakageFreePreprocessor
from phase1_leakage_free.split_manager import GroupSplitManager, SplitManifest


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) across uniform confidence bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def compute_precision_recall_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pair_provenances: List[Any],
    k_list: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """Compute anchor-level Precision@K and Recall@K grouped by fragment A."""
    anchors: Dict[str, List[Tuple[float, int]]] = {}
    for i, prov in enumerate(pair_provenances):
        anc = prov.frag_a_id
        anchors.setdefault(anc, []).append((float(y_prob[i]), int(y_true[i])))

    results = {}
    for k in k_list:
        p_at_k = []
        r_at_k = []
        for anc, candidates in anchors.items():
            total_pos = sum(c[1] for c in candidates)
            if total_pos == 0:
                continue
            # Sort candidates by predicted score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_k = candidates[:k]
            top_k_pos = sum(c[1] for c in top_k)
            p_at_k.append(top_k_pos / len(top_k))
            r_at_k.append(top_k_pos / total_pos)

        results[f"precision@{k}"] = float(np.mean(p_at_k)) if p_at_k else 0.0
        results[f"recall@{k}"] = float(np.mean(r_at_k)) if r_at_k else 0.0

    return results


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pair_provenances: List[Any],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute the full suite of forensic performance and calibration metrics."""
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    pos_ratio = float(n_pos / len(y_true)) if len(y_true) > 0 else 0.0

    # Handle single-class edge cases gracefully
    if n_pos == 0 or n_neg == 0:
        roc_auc = 0.5
        pr_auc = pos_ratio
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    at_k = compute_precision_recall_at_k(y_true, y_prob, pair_provenances)

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(loss, 4),
        "ece": round(ece, 4),
        "precision@1": round(at_k.get("precision@1", 0.0), 4),
        "recall@1": round(at_k.get("recall@1", 0.0), 4),
        "precision@5": round(at_k.get("precision@5", 0.0), 4),
        "recall@5": round(at_k.get("recall@5", 0.0), 4),
        "confusion_matrix": cm,
        "class_balance": {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "total": len(y_true),
            "pos_ratio": round(pos_ratio, 4),
        },
    }


def run_baseline_pipeline(
    seed: int = SEED,
    context_mode: str = "metadata_independent",
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete leakage-free training, artifact storage, and evaluation."""
    start_time = time.time()
    set_seeds(seed)

    base_dir = Path(__file__).resolve().parent
    artifacts_dir = artifacts_dir or (base_dir / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing GroupSplitManager...")
    mgr = GroupSplitManager(seed=seed)
    manifest = mgr.create_splits(val_file_count=5, cal_file_count=4)
    split_manifest_path = base_dir / "split_manifest.json"
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    train_files = set(manifest.train_files)
    val_files = set(manifest.val_files)
    cal_files = set(manifest.cal_files)
    test_files = set(manifest.test_files)

    logger.info("Splits created: %d train, %d val, %d cal, %d test files.",
                len(train_files), len(val_files), len(cal_files), len(test_files))

    # Load ground-truth fragments and binary blobs
    with open(GROUND_TRUTH_DIR / "fragments.csv", newline="", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    meta_rows.sort(key=lambda r: int(r["fragment_id"][1:]))

    all_fragments: List[GroundTruthFragment] = []
    for r in meta_rows:
        blob_path = FRAGMENTS_DIR / "blobs" / f"{r['fragment_id']}.bin"
        b_bytes = blob_path.read_bytes()
        if len(b_bytes) < FRAGMENT_SIZE:
            b_bytes = b_bytes.ljust(FRAGMENT_SIZE, b"\x00")

        # In metadata_independent mode, observed_offset is 0, allocation_status is unallocated
        obs = ObservableFragment(
            fragment_id=r["fragment_id"],
            observed_offset=0,
            fragment_bytes=b_bytes,
            fragment_length=len(b_bytes),
            allocation_status="unallocated",
            filesystem_metadata={"fs_type": "RAW"},
            filesystem_region="data_area",
            acquisition_source="carver",
        )
        gt = GroundTruthFragment(
            observable=obs,
            source_file_id=r["source_file"],
            logical_offset=int(r["offset_in_source"]),
            logical_sequence_index=int(r["original_position"]),
            original_fragment_id=r["fragment_id"],
            original_bytes=b_bytes,
            is_corrupted=False,
            is_missing=False,
            similar_group=r["similar_group"],
            split="train" if r["source_file"] in train_files else (
                "val" if r["source_file"] in val_files else (
                    "cal" if r["source_file"] in cal_files else "test"
                )
            ),
        )
        all_fragments.append(gt)

    logger.info("Loaded %d fragments across %d files.", len(all_fragments), len(manifest.file_metadata))

    # Build leakage-free feature engine exclusively over observable fragment views
    logger.info("Building LeakageFreeFeatureEngine (context_mode=%s)...", context_mode)
    engine = LeakageFreeFeatureEngine(
        [f.to_observable() for f in all_fragments],
        context_mode=context_mode,
    )

    # Generate isolated pair datasets
    logger.info("Generating pairwise datasets per partition...")
    pair_gen = PairGeneratorV2(seed=seed)
    train_ds = pair_gen.generate_pairs_for_partition(all_fragments, train_files, engine, "train", seed_offset=0)
    val_ds = pair_gen.generate_pairs_for_partition(all_fragments, val_files, engine, "val", seed_offset=10)
    cal_ds = pair_gen.generate_pairs_for_partition(all_fragments, cal_files, engine, "cal", seed_offset=20)
    test_ds = pair_gen.generate_pairs_for_partition(all_fragments, test_files, engine, "test", seed_offset=30)

    logger.info("Pairs generated: %d train, %d val, %d cal, %d test.",
                len(train_ds), len(val_ds), len(cal_ds), len(test_ds))

    # Preprocessing strictly fitted on train_ds.X
    logger.info("Fitting LeakageFreePreprocessor on training data...")
    preprocessor = LeakageFreePreprocessor(scaler_type="standard")
    preprocessor.fit(train_ds.X, feature_names=train_ds.feature_names)
    preprocessor_path = artifacts_dir / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)

    X_train_norm = preprocessor.transform(train_ds.X)
    X_val_norm = preprocessor.transform(val_ds.X)
    X_cal_norm = preprocessor.transform(cal_ds.X)
    X_test_norm = preprocessor.transform(test_ds.X)

    results: Dict[str, Any] = {
        "metadata": {
            "version": "2.0.0-leakage_free",
            "context_mode": context_mode,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_train_pairs": len(train_ds),
            "n_val_pairs": len(val_ds),
            "n_cal_pairs": len(cal_ds),
            "n_test_pairs": len(test_ds),
        },
        "same_file_task": {},
        "next_fragment_task": {},
        "deep_metric_learning": {},
    }

    # -------------------------------------------------------------
    # 1. Supervised Learning: Same-File Prediction (y_same)
    # -------------------------------------------------------------
    logger.info(">>> Training Task 1: Same-File Classifiers (y_same)...")
    task1_models = {
        "random_forest": RandomForestModel(n_estimators=100, seed=seed),
        "xgboost": XGBoostModel(n_estimators=100, seed=seed),
        "mlp_torch": MLPTorchModel(epochs=25, seed=seed),
        "logistic_regression": LogisticModel(seed=seed),
    }

    for name, model in task1_models.items():
        logger.info("Fitting same-file %s...", name)
        model.fit(X_train_norm, train_ds.y_same, X_val=X_val_norm, y_val=val_ds.y_same)
        save_path = artifacts_dir / f"same_{name}.{'joblib' if hasattr(model, 'save') and name != 'mlp_torch' else 'pt'}"
        model.save(save_path)

        test_probs = model.predict_proba(X_test_norm)
        eval_metrics = evaluate_predictions(test_ds.y_same, test_probs, test_ds.provenance)
        results["same_file_task"][name] = eval_metrics
        logger.info("Same-file %s: ROC-AUC = %.4f, PR-AUC = %.4f, F1 = %.4f",
                    name, eval_metrics["roc_auc"], eval_metrics["pr_auc"], eval_metrics["f1"])

    # -------------------------------------------------------------
    # 2. Supervised Learning: Next-Fragment Prediction (y_next)
    # -------------------------------------------------------------
    logger.info(">>> Training Task 2: Next-Fragment Classifiers (y_next)...")
    task2_models = {
        "random_forest": RandomForestModel(n_estimators=100, seed=seed),
        "xgboost": XGBoostModel(n_estimators=100, seed=seed),
        "mlp_torch": MLPTorchModel(epochs=25, seed=seed),
        "logistic_regression": LogisticModel(seed=seed),
    }

    for name, model in task2_models.items():
        logger.info("Fitting next-fragment %s...", name)
        model.fit(X_train_norm, train_ds.y_next, X_val=X_val_norm, y_val=val_ds.y_next)
        save_path = artifacts_dir / f"next_{name}.{'joblib' if hasattr(model, 'save') and name != 'mlp_torch' else 'pt'}"
        model.save(save_path)

        test_probs = model.predict_proba(X_test_norm)
        eval_metrics = evaluate_predictions(test_ds.y_next, test_probs, test_ds.provenance)
        results["next_fragment_task"][name] = eval_metrics
        logger.info("Next-fragment %s: ROC-AUC = %.4f, PR-AUC = %.4f, F1 = %.4f",
                    name, eval_metrics["roc_auc"], eval_metrics["pr_auc"], eval_metrics["f1"])

    # -------------------------------------------------------------
    # 3. Deep Metric Learning: Siamese MLP and Siamese CNN
    # -------------------------------------------------------------
    logger.info(">>> Training Deep Metric Learning Encoders...")
    train_frag_indices = [i for i, f in enumerate(all_fragments) if f.source_file_id in train_files]
    test_frag_indices = [i for i, f in enumerate(all_fragments) if f.source_file_id in test_files]

    # Map pairs to relative training indices for Siamese fit
    train_single_matrix = engine.single_matrix_
    train_raw_matrix = engine.data_

    # Siamese MLP (Single handcrafted features -> 32d embedding)
    siamese_mlp = SiameseNetwork(input_dim=train_single_matrix.shape[1], epochs=15, seed=seed)
    siamese_mlp.fit(
        single_matrix=train_single_matrix,
        pairs=train_ds.pair_idx,
        labels_same=train_ds.y_same,
        labels_next=train_ds.y_next,
    )
    siamese_mlp_path = artifacts_dir / "siamese_mlp.pt"
    siamese_mlp.save(siamese_mlp_path)

    # Evaluate Siamese MLP on test pairs
    test_embs_mlp = siamese_mlp.embed(engine.single_matrix_)
    mlp_pair_sims = (test_embs_mlp[test_ds.pair_idx[:, 0]] * test_embs_mlp[test_ds.pair_idx[:, 1]]).sum(axis=1)
    # Cosine sim in [-1, 1] mapped to [0, 1] for probability-like scoring
    mlp_pair_probs = np.clip((mlp_pair_sims + 1.0) / 2.0, 0.0, 1.0)
    results["deep_metric_learning"]["siamese_mlp_same"] = evaluate_predictions(
        test_ds.y_same, mlp_pair_probs, test_ds.provenance
    )
    results["deep_metric_learning"]["siamese_mlp_next"] = evaluate_predictions(
        test_ds.y_next, mlp_pair_probs, test_ds.provenance
    )

    # Siamese CNN (Raw byte payloads -> 32d embedding)
    siamese_cnn = SiameseCNN(block_size=FRAGMENT_SIZE, epochs=15, seed=seed)
    siamese_cnn.fit(
        raw_matrix=train_raw_matrix,
        pairs=train_ds.pair_idx,
        labels_same=train_ds.y_same,
        labels_next=train_ds.y_next,
    )
    siamese_cnn_path = artifacts_dir / "siamese_cnn.pt"
    siamese_cnn.save(siamese_cnn_path)

    # Evaluate Siamese CNN on test pairs
    test_embs_cnn = siamese_cnn.embed(engine.data_)
    cnn_pair_sims = (test_embs_cnn[test_ds.pair_idx[:, 0]] * test_embs_cnn[test_ds.pair_idx[:, 1]]).sum(axis=1)
    cnn_pair_probs = np.clip((cnn_pair_sims + 1.0) / 2.0, 0.0, 1.0)
    results["deep_metric_learning"]["siamese_cnn_same"] = evaluate_predictions(
        test_ds.y_same, cnn_pair_probs, test_ds.provenance
    )
    results["deep_metric_learning"]["siamese_cnn_next"] = evaluate_predictions(
        test_ds.y_next, cnn_pair_probs, test_ds.provenance
    )

    # Output baseline results JSON
    results_path = base_dir / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote baseline evaluation results to %s", results_path)

    # Build experiment manifest
    duration = time.time() - start_time
    artifact_files = sorted(artifacts_dir.glob("*.*"))
    artifact_hashes = {f.name: compute_file_hash(f) for f in artifact_files}

    manifest_data = {
        "experiment_id": f"phase1_baseline_leakage_free_{seed}",
        "version": "2.0.0-leakage_free",
        "timestamp": results["metadata"]["timestamp"],
        "duration_seconds": round(duration, 2),
        "seed": seed,
        "context_mode": context_mode,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "split_manifest_hash": manifest.manifest_hash,
        "splits": manifest.to_dict()["counts"],
        "artifact_hashes": artifact_hashes,
        "summary_metrics": {
            "same_file_rf_auc": results["same_file_task"]["random_forest"]["roc_auc"],
            "same_file_xgb_auc": results["same_file_task"]["xgboost"]["roc_auc"],
            "next_frag_rf_auc": results["next_fragment_task"]["random_forest"]["roc_auc"],
            "next_frag_xgb_auc": results["next_fragment_task"]["xgboost"]["roc_auc"],
            "siamese_mlp_same_auc": results["deep_metric_learning"]["siamese_mlp_same"]["roc_auc"],
            "siamese_cnn_same_auc": results["deep_metric_learning"]["siamese_cnn_same"]["roc_auc"],
        },
    }

    manifest_path = base_dir / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Wrote experiment manifest to %s", manifest_path)

    return results


if __name__ == "__main__":
    run_baseline_pipeline()
