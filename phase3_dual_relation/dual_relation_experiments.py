"""Dual-Relation Controlled Experimental Benchmark (v2.0.0-phase3).

Conducts rigorous experimental comparison across 4 architectural variants:
- Variant A: Next-fragment only (replicates v1 baseline behavior, omitting same-file model)
- Variant B: Same-file only (clustering only, no directed sequence modeling)
- Variant C: Independent dual-model inference (SameFileModelV2 + NextFragmentModelV2)
- Variant D: Shared-encoder multi-task inference (MultiTaskDualModel)

Evaluates:
1. Pair-level discrimination: ROC-AUC, PR-AUC, Avg Precision, P@K, R@K, ECE, Brier
2. Hard negative resilience: Same-file non-adjacent pairs, same-format negatives, cross-file noise
3. Reconstruction-level performance via EvaluationEngine:
   - Exact byte recovery rate (SHA256 match)
   - Partial recovery rate (LCS >= 0.5)
   - Fragment placement accuracy
   - False file merge rate
   - False file split rate
   - Reconstruction runtime & candidate graph size
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import logger, set_seeds
from config import CORPUS_PLAN, FRAGMENT_SIZE, FRAGMENTS_DIR, GROUND_TRUTH_DIR, SEED
from phase1_leakage_free.baseline_training_v2 import compute_ece, compute_precision_recall_at_k
from phase1_leakage_free.fragment_schema_v2 import GroundTruthFragment, ObservableFragment
from phase1_leakage_free.leakage_free_features import LeakageFreeFeatureEngine
from phase1_leakage_free.pair_generator_v2 import PairDatasetV2, PairGeneratorV2
from phase1_leakage_free.preprocessing_v2 import LeakageFreePreprocessor
from phase1_leakage_free.split_manager import SplitManifest
from phase2_dataset_evaluation.evaluation_engine import EvaluationEngine, ReconstructedFile, sha256_bytes
from phase3_dual_relation.dual_relation_graph import (
    DualRelationReconstructionEngine,
    build_dual_relation_graph,
)
from phase3_dual_relation.dual_relation_inference import DualRelationInferenceEngine
from phase3_dual_relation.multitask_model import MultiTaskDualModel
from phase3_dual_relation.next_fragment_model_v2 import NextFragmentModelV2
from phase3_dual_relation.same_file_model_v2 import SameFileModelV2


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on validation set."""
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.70, 66):
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return float(best_t)


def evaluate_pair_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    provenance: List[Any],
    task_name: str,
    threshold: float = 0.5,
    optimal_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute complete discrimination and calibration metrics for a pairwise task."""
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    y_pred_default = (y_prob >= threshold).astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        roc_auc = 0.5
        pr_auc = float(n_pos / len(y_true)) if len(y_true) > 0 else 0.0
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    prec = float(precision_score(y_true, y_pred_default, zero_division=0))
    rec = float(recall_score(y_true, y_pred_default, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_default, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(compute_ece(y_true, y_prob))

    at_k = compute_precision_recall_at_k(y_true, y_prob, provenance)

    opt_metrics = {}
    if optimal_threshold is not None:
        y_pred_opt = (y_prob >= optimal_threshold).astype(int)
        opt_metrics = {
            "optimal_threshold": round(optimal_threshold, 4),
            "opt_precision": round(float(precision_score(y_true, y_pred_opt, zero_division=0)), 4),
            "opt_recall": round(float(recall_score(y_true, y_pred_opt, zero_division=0)), 4),
            "opt_f1": round(float(f1_score(y_true, y_pred_opt, zero_division=0)), 4),
        }

    res = {
        "task": task_name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_at_05": round(prec, 4),
        "recall_at_05": round(rec, 4),
        "f1_at_05": round(f1, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "precision@1": round(at_k.get("precision@1", 0.0), 4),
        "precision@5": round(at_k.get("precision@5", 0.0), 4),
        "recall@1": round(at_k.get("recall@1", 0.0), 4),
        "recall@5": round(at_k.get("recall@5", 0.0), 4),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    res.update(opt_metrics)
    return res


def evaluate_hard_negatives(
    test_ds: PairDatasetV2,
    p_same: np.ndarray,
    p_next: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate resilience against hard negative sub-populations."""
    strat_masks: Dict[str, List[int]] = {}
    for idx, prov in enumerate(test_ds.provenance):
        strat_masks.setdefault(prov.strategy, []).append(idx)

    res = {}
    for strat, indices in strat_masks.items():
        sub_indices = np.array(indices)
        sub_y_same = test_ds.y_same[sub_indices]
        sub_y_next = test_ds.y_next[sub_indices]
        sub_p_same = p_same[sub_indices]
        sub_p_next = p_next[sub_indices]

        # Calculate mean scores and false positive rates
        res[strat] = {
            "count": len(sub_indices),
            "mean_p_same": round(float(np.mean(sub_p_same)), 4),
            "mean_p_next": round(float(np.mean(sub_p_next)), 4),
            "fpr_same": round(float(np.mean(sub_p_same[sub_y_same == 0] >= 0.5)), 4) if np.any(sub_y_same == 0) else 0.0,
            "fpr_next": round(float(np.mean(sub_p_next[sub_y_next == 0] >= 0.5)), 4) if np.any(sub_y_next == 0) else 0.0,
        }

    return res


def run_dual_relation_benchmark(seed: int = SEED) -> Dict[str, Any]:
    """Execute end-to-end benchmark across variants A, B, C, D."""
    start_time = time.time()
    set_seeds(seed)

    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load split manifest created in Phase 1
    split_manifest_path = PROJECT_ROOT / Path("phase1_leakage_free/split_manifest.json")
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest at {split_manifest_path}")

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    train_files = set(manifest_data["train_files"])
    val_files = set(manifest_data["val_files"])
    cal_files = set(manifest_data["cal_files"])
    test_files = set(manifest_data["test_files"])

    logger.info("Loaded split manifest: %d train, %d val, %d cal, %d test files.",
                len(train_files), len(val_files), len(cal_files), len(test_files))

    # 2. Load all ground-truth fragments and blobs
    with open(GROUND_TRUTH_DIR / "fragments.csv", newline="", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    meta_rows.sort(key=lambda r: int(r["fragment_id"][1:]))

    all_fragments: List[GroundTruthFragment] = []
    file_bytes_map: Dict[str, List[bytes]] = {}
    file_metadata: Dict[str, Dict[str, Any]] = {}

    for r in meta_rows:
        blob_path = FRAGMENTS_DIR / "blobs" / f"{r['fragment_id']}.bin"
        b_bytes = blob_path.read_bytes()
        if len(b_bytes) < FRAGMENT_SIZE:
            b_bytes = b_bytes.ljust(FRAGMENT_SIZE, b"\x00")

        fn = r["source_file"]
        file_bytes_map.setdefault(fn, []).append(b_bytes)

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
            source_file_id=fn,
            logical_offset=int(r["offset_in_source"]),
            logical_sequence_index=int(r["original_position"]),
            original_fragment_id=r["fragment_id"],
            original_bytes=b_bytes,
            is_corrupted=False,
            is_missing=False,
            similar_group=r["similar_group"],
            split="train" if fn in train_files else (
                "val" if fn in val_files else ("cal" if fn in cal_files else "test")
            ),
        )
        all_fragments.append(gt)

    # Compute ground truth hashes and sizes for test files
    for fn in test_files:
        raw_full = b"".join(file_bytes_map[fn])
        file_metadata[fn] = {
            "source_file_id": fn,
            "original_file_hash": sha256_bytes(raw_full),
            "file_size_bytes": len(raw_full),
            "total_fragments": len(file_bytes_map[fn]),
            "file_type": fn.split(".")[-1].upper(),
        }

    # 3. Build Leakage-Free Feature Engine
    logger.info("Extracting leakage-free features across all fragments...")
    engine = LeakageFreeFeatureEngine(
        [f.to_observable() for f in all_fragments],
        context_mode="metadata_independent",
    )

    # 4. Generate pairwise datasets per partition
    logger.info("Generating pairwise datasets per partition...")
    pair_gen = PairGeneratorV2(seed=seed)
    train_ds = pair_gen.generate_pairs_for_partition(all_fragments, train_files, engine, "train", seed_offset=0)
    val_ds = pair_gen.generate_pairs_for_partition(all_fragments, val_files, engine, "val", seed_offset=10)
    cal_ds = pair_gen.generate_pairs_for_partition(all_fragments, cal_files, engine, "cal", seed_offset=20)
    test_ds = pair_gen.generate_pairs_for_partition(all_fragments, test_files, engine, "test", seed_offset=30)

    # 5. Fit preprocessor strictly on train_ds.X
    logger.info("Fitting LeakageFreePreprocessor...")
    preprocessor = LeakageFreePreprocessor(scaler_type="standard")
    preprocessor.fit(train_ds.X, feature_names=train_ds.feature_names)

    X_train_fwd = preprocessor.transform(train_ds.X)
    X_val_fwd = preprocessor.transform(val_ds.X)
    X_cal_fwd = preprocessor.transform(cal_ds.X)
    X_test_fwd = preprocessor.transform(test_ds.X)

    # Compute reverse feature matrices for exact symmetry and bidirectional inference
    logger.info("Computing reverse directional features...")
    X_train_rev = preprocessor.transform(engine.pair_matrix_vec(np.column_stack([train_ds.pair_idx[:, 1], train_ds.pair_idx[:, 0]])))
    X_val_rev = preprocessor.transform(engine.pair_matrix_vec(np.column_stack([val_ds.pair_idx[:, 1], val_ds.pair_idx[:, 0]])))
    X_cal_rev = preprocessor.transform(engine.pair_matrix_vec(np.column_stack([cal_ds.pair_idx[:, 1], cal_ds.pair_idx[:, 0]])))
    X_test_rev = preprocessor.transform(engine.pair_matrix_vec(np.column_stack([test_ds.pair_idx[:, 1], test_ds.pair_idx[:, 0]])))

    # -----------------------------------------------------------------
    # Model Training & Calibration
    # -----------------------------------------------------------------
    logger.info(">>> Training Independent Model 1: SameFileModelV2...")
    same_model = SameFileModelV2(model_type="random_forest", n_estimators=120, max_depth=16, seed=seed)
    same_model.fit(X_train_fwd, train_ds.y_same, X_cal=X_cal_fwd, y_cal_same=cal_ds.y_same)
    same_model.save(artifacts_dir / "same_file_rf.joblib")

    logger.info(">>> Training Independent Model 2: NextFragmentModelV2...")
    next_model = NextFragmentModelV2(model_type="random_forest", n_estimators=120, max_depth=16, seed=seed)
    next_model.fit(X_train_fwd, train_ds.y_next, X_cal=X_cal_fwd, y_cal_next=cal_ds.y_next)
    next_model.save(artifacts_dir / "next_fragment_rf.joblib")

    logger.info(">>> Training Multi-Task Model: MultiTaskDualModel...")
    multitask_model = MultiTaskDualModel(
        input_dim=X_train_fwd.shape[1],
        shared_dim=128,
        head_hidden=64,
        lambda_same=1.0,
        lambda_next=1.0,
        epochs=30,
        seed=seed,
    )
    multitask_model.fit(
        X_train_fwd=X_train_fwd,
        y_train_same=train_ds.y_same,
        y_train_next=train_ds.y_next,
        X_train_rev=X_train_rev,
        X_val_fwd=X_val_fwd,
        y_val_same=val_ds.y_same,
        y_val_next=val_ds.y_next,
        X_val_rev=X_val_rev,
        X_cal_fwd=X_cal_fwd,
        y_cal_same=cal_ds.y_same,
        y_cal_next=cal_ds.y_next,
        X_cal_rev=X_cal_rev,
    )
    multitask_model.save(artifacts_dir / "multitask_dual.pt")

    # -----------------------------------------------------------------
    # Validation Threshold Tuning & Pair-Level Evaluation
    # -----------------------------------------------------------------
    logger.info(">>> Tuning optimal task thresholds on validation partition...")
    p_same_val_indep, _ = same_model.predict_proba_symmetric(X_val_fwd, X_val_rev)
    p_next_val_indep, _, _, _ = next_model.predict_proba_directed(X_val_fwd, X_val_rev)
    opt_tau_same_indep = find_optimal_threshold(val_ds.y_same, p_same_val_indep)
    opt_tau_next_indep = find_optimal_threshold(val_ds.y_next, p_next_val_indep)

    p_same_val_mt, p_next_val_mt, _ = multitask_model.predict_proba(X_val_fwd, X_val_rev)
    opt_tau_same_mt = find_optimal_threshold(val_ds.y_same, p_same_val_mt)
    opt_tau_next_mt = find_optimal_threshold(val_ds.y_next, p_next_val_mt)

    logger.info("Validation optimal thresholds: indep (same=%.4f, next=%.4f), mt (same=%.4f, next=%.4f)",
                opt_tau_same_indep, opt_tau_next_indep, opt_tau_same_mt, opt_tau_next_mt)

    logger.info(">>> Evaluating pair-level discrimination on held-out test split...")
    # 1. Independent models
    p_same_indep_cal, _ = same_model.predict_proba_symmetric(X_test_fwd, X_test_rev)
    p_next_indep_cal, p_prev_indep_cal, _, _ = next_model.predict_proba_directed(X_test_fwd, X_test_rev)

    # 2. Multi-task model
    p_same_mt_cal, p_next_mt_cal, p_prev_mt_cal = multitask_model.predict_proba(X_test_fwd, X_test_rev)

    pair_results = {
        "independent_models": {
            "same_file_task": evaluate_pair_performance(
                test_ds.y_same, p_same_indep_cal, test_ds.provenance, "same_file", optimal_threshold=opt_tau_same_indep
            ),
            "next_fragment_task": evaluate_pair_performance(
                test_ds.y_next, p_next_indep_cal, test_ds.provenance, "next_fragment", optimal_threshold=opt_tau_next_indep
            ),
            "hard_negatives": evaluate_hard_negatives(test_ds, p_same_indep_cal, p_next_indep_cal),
        },
        "multitask_model": {
            "same_file_task": evaluate_pair_performance(
                test_ds.y_same, p_same_mt_cal, test_ds.provenance, "same_file", optimal_threshold=opt_tau_same_mt
            ),
            "next_fragment_task": evaluate_pair_performance(
                test_ds.y_next, p_next_mt_cal, test_ds.provenance, "next_fragment", optimal_threshold=opt_tau_next_mt
            ),
            "hard_negatives": evaluate_hard_negatives(test_ds, p_same_mt_cal, p_next_mt_cal),
        },
    }

    # -----------------------------------------------------------------
    # Reconstruction-Level Evaluation across Variants A, B, C, D
    # -----------------------------------------------------------------
    logger.info(">>> Conducting reconstruction-level evaluation across Variants A, B, C, D...")
    eval_engine = EvaluationEngine()

    # Build canonical test pool
    test_fragments = [f for f in all_fragments if f.source_file_id in test_files]
    test_pool_map: Dict[str, bytes] = {}
    test_canonical_ids: List[str] = []
    local_to_engine_idx: Dict[int, int] = {}

    for i, tf in enumerate(test_fragments):
        cid = f"{tf.source_file_id}_frag_{tf.logical_sequence_index:04d}"
        test_canonical_ids.append(cid)
        test_pool_map[cid] = tf.original_bytes
        # Map local index to engine index
        eng_idx = int(tf.original_fragment_id[1:])
        local_to_engine_idx[i] = eng_idx

    n_test_pool = len(test_fragments)
    logger.info("Test pool contains %d fragments across %d files.", n_test_pool, len(test_files))

    # Generate candidate test successor pairs using KNN-like histogram similarity + probes
    logger.info("Generating candidate successor transitions for test pool...")
    test_eng_indices = np.array([local_to_engine_idx[i] for i in range(n_test_pool)], dtype=np.int64)
    hists = engine.cache_["byte"]["hist"][test_eng_indices]
    norms = np.linalg.norm(hists, axis=1, keepdims=True) + 1e-7
    hists_norm = hists / norms
    sim_matrix = hists_norm @ hists_norm.T
    np.fill_diagonal(sim_matrix, -1.0)

    candidate_pairs_set: Set[Tuple[int, int]] = set()
    top_k_cand = min(16, n_test_pool - 1)
    rng = np.random.RandomState(seed)

    for a in range(n_test_pool):
        top_indices = np.argsort(-sim_matrix[a])[:top_k_cand]
        for b in top_indices:
            candidate_pairs_set.add((a, int(b)))
        for _ in range(3):
            rnd_b = rng.randint(0, n_test_pool)
            if rnd_b != a:
                candidate_pairs_set.add((a, int(rnd_b)))

    test_pairs_arr = np.array(list(candidate_pairs_set), dtype=np.int64)
    # Map to engine indices
    eng_pairs_fwd = np.column_stack([
        [local_to_engine_idx[a] for a in test_pairs_arr[:, 0]],
        [local_to_engine_idx[b] for a, b in test_pairs_arr],
    ])
    eng_pairs_rev = np.column_stack([eng_pairs_fwd[:, 1], eng_pairs_fwd[:, 0]])

    logger.info("Extracting pair features for %d candidate test transitions...", len(eng_pairs_fwd))
    X_pool_fwd = preprocessor.transform(engine.pair_matrix_vec(eng_pairs_fwd))
    X_pool_rev = preprocessor.transform(engine.pair_matrix_vec(eng_pairs_rev))

    # Pre-compute probabilities for all pool candidate transitions
    p_same_pool_indep, _ = same_model.predict_proba_symmetric(X_pool_fwd, X_pool_rev)
    p_next_pool_indep, p_prev_pool_indep, _, _ = next_model.predict_proba_directed(X_pool_fwd, X_pool_rev)

    p_same_pool_mt, p_next_pool_mt, p_prev_pool_mt = multitask_model.predict_proba(X_pool_fwd, X_pool_rev)

    gt_manifest = {"files": file_metadata}
    test_pool_id_set = set(test_canonical_ids)

    recon_results: Dict[str, Any] = {}

    variants = [
        ("Variant_A_Next_Only", "variant_a_next_only", p_next_pool_indep, p_next_pool_indep, p_prev_pool_indep, opt_tau_same_indep, opt_tau_next_indep),
        ("Variant_B_Same_Only", "variant_b_same_only", p_same_pool_indep, p_next_pool_indep, p_prev_pool_indep, opt_tau_same_indep, opt_tau_next_indep),
        ("Variant_C_Dual_Independent", "variant_c_dual_independent", p_same_pool_indep, p_next_pool_indep, p_prev_pool_indep, opt_tau_same_indep, opt_tau_next_indep),
        ("Variant_D_Shared_MultiTask", "variant_c_dual_independent", p_same_pool_mt, p_next_pool_mt, p_prev_pool_mt, opt_tau_same_mt, opt_tau_next_mt),
    ]

    for label, mode_str, p_s_arr, p_n_arr, p_p_arr, tau_s, tau_n in variants:
        t0 = time.time()
        recon_engine = DualRelationReconstructionEngine(tau_same=tau_s, tau_next=tau_n, beam_width=5)
        # Build candidate graph
        g = build_dual_relation_graph(
            fragment_ids=test_canonical_ids,
            successor_pairs=test_pairs_arr,
            p_same=p_s_arr,
            p_next=p_n_arr,
            p_prev=p_p_arr,
            top_k=8,
            model_version=f"phase3_{label.lower()}",
        )
        recons = recon_engine.reconstruct(g, test_pool_map, mode=mode_str)
        elapsed = time.time() - t0

        eff_stats = {
            "runtime_seconds": round(elapsed, 4),
            "candidate_graph_nodes": g.number_of_nodes(),
            "candidate_graph_edges": g.number_of_edges(),
        }

        eval_res = eval_engine.evaluate_reconstruction(
            ground_truth_manifest=gt_manifest,
            reconstructed_files=recons,
            all_pool_fragment_ids=test_pool_id_set,
            efficiency_stats=eff_stats,
        )

        recon_results[label] = {
            "mode": mode_str,
            "metrics": eval_res,
        }
        logger.info("%s: Exact Recovery = %.4f, Partial = %.4f, Merge Rate = %.4f, Runtime = %.3fs",
                    label,
                    eval_res["file_level"]["exact_reconstruction_rate"],
                    eval_res["file_level"]["partial_reconstruction_rate"],
                    eval_res["file_level"]["false_file_merge_rate"],
                    elapsed)

    total_benchmark_time = time.time() - start_time

    # -----------------------------------------------------------------
    # Save Model Registry & Results
    # -----------------------------------------------------------------
    registry = {
        "version": "2.0.0-phase3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {
            "same_file_model_v2": {
                "architecture": "RandomForestClassifier",
                "n_estimators": 120,
                "max_depth": 16,
                "calibration": "sigmoid (Platt scaling)",
                "artifact_path": str(artifacts_dir / "same_file_rf.joblib"),
                "symmetry_guarantee": "Permutation invariant via 0.5 * (fwd + rev)",
            },
            "next_fragment_model_v2": {
                "architecture": "RandomForestClassifier",
                "n_estimators": 120,
                "max_depth": 16,
                "calibration": "sigmoid (Platt scaling)",
                "artifact_path": str(artifacts_dir / "next_fragment_rf.joblib"),
                "directionality": "Asymmetric directed transition + reverse consensus",
            },
            "multitask_dual_model": {
                "architecture": "PyTorch MultiTaskNet (Shared 128D Trunk + Dual Task Heads)",
                "shared_dim": 128,
                "head_hidden": 64,
                "lambda_same": 1.0,
                "lambda_next": 1.0,
                "calibration": "Platt scaling per head on calibration split",
                "artifact_path": str(artifacts_dir / "multitask_dual.pt"),
            },
        },
    }

    with open(base_dir / "model_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    results_data = {
        "metadata": {
            "version": "2.0.0-phase3",
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark_runtime_seconds": round(total_benchmark_time, 2),
            "n_train_files": len(train_files),
            "n_val_files": len(val_files),
            "n_cal_files": len(cal_files),
            "n_test_files": len(test_files),
        },
        "pair_level_benchmark": pair_results,
        "reconstruction_benchmark": recon_results,
    }

    with open(base_dir / "dual_relation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    logger.info("Phase 3 benchmark complete! Results written to dual_relation_results.json.")
    return results_data


if __name__ == "__main__":
    run_dual_relation_benchmark()
