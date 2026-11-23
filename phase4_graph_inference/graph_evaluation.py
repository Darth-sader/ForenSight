"""Comprehensive Forensic Graph Evaluation Suite (v2.0.0-phase4).

Evaluates:
1. Candidate Generation & Recall@K across Fixed K (4, 8, 16) and Adaptive K.
2. Graph Consistency, Cycles, and Topological Conflict Diagnostics.
3. Relational GNN Refinement: Baseline vs GraphSAGE vs GAT.
4. Multi-Hypothesis Beam Search Reconstruction across Beam Widths (W=1, 3, 5, 10).
5. Emits complete results into phase4_results.json.
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
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from common import logger, set_seeds
from config import CORPUS_PLAN, FRAGMENT_SIZE, FRAGMENTS_DIR, GROUND_TRUTH_DIR, SEED
from phase1_leakage_free.fragment_schema_v2 import GroundTruthFragment, ObservableFragment
from phase1_leakage_free.leakage_free_features import LeakageFreeFeatureEngine
from phase1_leakage_free.pair_generator_v2 import PairGeneratorV2
from phase1_leakage_free.preprocessing_v2 import LeakageFreePreprocessor
from phase2_dataset_evaluation.evaluation_engine import EvaluationEngine, ReconstructedFile, sha256_bytes
from phase3_dual_relation.next_fragment_model_v2 import NextFragmentModelV2
from phase3_dual_relation.same_file_model_v2 import SameFileModelV2
from phase4_graph_inference.adaptive_candidate_generator import AdaptiveCandidateGenerator
from phase4_graph_inference.beam_search_v2 import MultiHypothesisBeamSearchV2
from phase4_graph_inference.candidate_generator_v2 import CandidateGeneratorV2
from phase4_graph_inference.gat_refinement import GATRefiner
from phase4_graph_inference.graph_consistency import GraphConsistencyChecker
from phase4_graph_inference.graph_diagnostics import GraphDiagnostics
from phase4_graph_inference.graphsage_refinement import GraphSAGERefiner


def run_phase4_benchmark(seed: int = SEED) -> Dict[str, Any]:
    """Execute complete Phase 4 experimental pipeline."""
    start_time = time.time()
    set_seeds(seed)

    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load split manifest
    split_manifest_path = PROJECT_ROOT / Path("phase1_leakage_free/split_manifest.json")
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
    gt_successors_all: Dict[str, str] = {}

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

    # Build true successor lookup per file (for Recall@K evaluation)
    for fn, chunks in file_bytes_map.items():
        frags_for_file = [f for f in all_fragments if f.source_file_id == fn]
        frags_for_file.sort(key=lambda x: x.logical_sequence_index)
        for i in range(len(frags_for_file) - 1):
            gt_successors_all[frags_for_file[i].observable.fragment_id] = frags_for_file[i + 1].observable.fragment_id

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

    # 3. Build LeakageFreeFeatureEngine
    logger.info("Extracting leakage-free features across all fragments...")
    engine = LeakageFreeFeatureEngine(
        [f.to_observable() for f in all_fragments],
        context_mode="metadata_independent",
    )

    test_gt_fragments = [f for f in all_fragments if f.source_file_id in test_files]
    test_obs_fragments = [f.to_observable() for f in test_gt_fragments]
    test_indices = np.array([int(f.original_fragment_id[1:]) for f in test_gt_fragments], dtype=np.int64)
    test_single_features = engine.single_matrix_[test_indices]
    test_gt_successors = {
        k: v for k, v in gt_successors_all.items() if k in {f.fragment_id for f in test_obs_fragments}
    }

    # Load Phase 3 trained models
    phase3_artifacts = PROJECT_ROOT / Path("phase3_dual_relation/artifacts")
    same_model = joblib.load(phase3_artifacts / "same_file_rf.joblib")
    next_model = joblib.load(phase3_artifacts / "next_fragment_rf.joblib")

    # Load preprocessor
    preprocessor = LeakageFreePreprocessor.load(PROJECT_ROOT / Path("phase1_leakage_free/artifacts/preprocessor.joblib"))

    # -------------------------------------------------------------
    # 1. Candidate Generation Benchmark: Fixed K vs Adaptive K
    # -------------------------------------------------------------
    logger.info(">>> 1. Benchmarking Candidate Generation (Fixed K=4, 8, 16 vs Adaptive K)...")
    cand_gen = CandidateGeneratorV2(use_embeddings=True, use_boundary=True)
    adaptive_gen = AdaptiveCandidateGenerator(k_min=4, k_base=8, k_max=16)

    cand_results = {}
    for k_val in [4, 8, 16]:
        g_k, pairs_k, metrics_k = cand_gen.generate_candidate_graph(
            test_obs_fragments,
            feature_matrix=test_single_features,
            fixed_k=k_val,
            true_successors=test_gt_successors,
        )
        cand_results[f"fixed_k_{k_val}"] = metrics_k.to_dict()
        logger.info("Fixed K=%d: Recall@1=%.4f, Recall@K=%.4f, Density=%.6f, Runtime=%.3fs",
                    k_val, metrics_k.recall_at_k.get(1, 0.0), metrics_k.recall_at_k.get(k_val, 0.0),
                    metrics_k.graph_density, metrics_k.generation_runtime_seconds)

    g_adapt, pairs_adapt, metrics_adapt, profiles = adaptive_gen.generate_adaptive_candidate_graph(
        test_obs_fragments,
        feature_matrix=test_single_features,
        true_successors=test_gt_successors,
    )
    cand_results["adaptive_k"] = metrics_adapt.to_dict()
    cand_results["adaptive_k"]["mean_allocated_k"] = round(float(np.mean([p.allocated_k for p in profiles])), 2)
    logger.info("Adaptive K: Mean-K=%.2f, Recall@1=%.4f, Recall@Max=%.4f, Density=%.6f, Runtime=%.3fs",
                cand_results["adaptive_k"]["mean_allocated_k"], metrics_adapt.recall_at_k.get(1, 0.0),
                metrics_adapt.recall_at_k.get(16, 0.0), metrics_adapt.graph_density, metrics_adapt.generation_runtime_seconds)

    # -------------------------------------------------------------
    # 2. Graph Consistency and Conflict Diagnostics
    # -------------------------------------------------------------
    logger.info(">>> 2. Auditing Graph Consistency and Conflict Diagnostics...")
    checker = GraphConsistencyChecker(tau_conflict=0.15)
    inconsistency_pre = checker.check_consistency(g_adapt)
    diagnostics_pre = GraphDiagnostics.diagnose_graph(g_adapt)

    # Apply Feedback Arc Set cycle repair
    g_repaired = checker.repair_feedback_arc_set(g_adapt)
    inconsistency_post = checker.check_consistency(g_repaired)
    diagnostics_post = GraphDiagnostics.diagnose_graph(g_repaired)

    consistency_results = {
        "pre_repair": {
            "inconsistency": inconsistency_pre.to_dict(),
            "diagnostics": diagnostics_pre.to_dict(),
        },
        "post_repair": {
            "inconsistency": inconsistency_post.to_dict(),
            "diagnostics": diagnostics_post.to_dict(),
        },
    }
    logger.info("Consistency Check: Pre-repair cycles=%d, Post-repair cycles=%d, Is DAG=%s",
                inconsistency_pre.cycle_count, inconsistency_post.cycle_count, diagnostics_post.is_dag)

    # -------------------------------------------------------------
    # 3. GNN Relational Refinement (GraphSAGE & GAT)
    # -------------------------------------------------------------
    logger.info(">>> 3. Training and Benchmarking GNN Refinement (GraphSAGE & GAT)...")
    # Build train candidate graph for GNN training (strict isolation)
    train_gt_frags = [f for f in all_fragments if f.source_file_id in train_files]
    train_obs_frags = [f.to_observable() for f in train_gt_frags]
    train_indices = np.array([int(f.original_fragment_id[1:]) for f in train_gt_frags], dtype=np.int64)
    train_single_features = engine.single_matrix_[train_indices]

    train_gt_successors = {
        k: v for k, v in gt_successors_all.items() if k in {f.fragment_id for f in train_obs_frags}
    }
    _, train_cand_pairs, _ = cand_gen.generate_candidate_graph(
        train_obs_frags,
        feature_matrix=train_single_features,
        fixed_k=8,
        true_successors=train_gt_successors,
    )

    # Compute training edge features & labels
    train_eng_u = train_indices[train_cand_pairs[:, 0]]
    train_eng_v = train_indices[train_cand_pairs[:, 1]]
    train_eng_pairs_fwd = np.column_stack([train_eng_u, train_eng_v])
    train_eng_pairs_rev = np.column_stack([train_eng_v, train_eng_u])

    X_train_cand_fwd = preprocessor.transform(engine.pair_matrix_vec(train_eng_pairs_fwd))
    X_train_cand_rev = preprocessor.transform(engine.pair_matrix_vec(train_eng_pairs_rev))

    p_same_train, _ = same_model.predict_proba_symmetric(X_train_cand_fwd, X_train_cand_rev)
    p_next_train, p_prev_train, _, _ = next_model.predict_proba_directed(X_train_cand_fwd, X_train_cand_rev)

    train_edge_features = np.column_stack([
        p_same_train,
        p_next_train,
        p_prev_train,
        X_train_cand_fwd[:, 0],  # byte cosine
        X_train_cand_fwd[:, 1],  # boundary
        np.zeros(len(train_cand_pairs)),  # physical context
    ]).astype(np.float32)

    # Ground truth labels for training candidate transitions
    y_next_train = np.array([
        1 if train_obs_frags[u].fragment_id in train_gt_successors and
             train_gt_successors[train_obs_frags[u].fragment_id] == train_obs_frags[v].fragment_id else 0
        for u, v in train_cand_pairs
    ], dtype=np.float32)

    y_same_train = np.array([
        1 if train_gt_frags[u].source_file_id == train_gt_frags[v].source_file_id else 0
        for u, v in train_cand_pairs
    ], dtype=np.float32)

    # Build test candidate edge features
    test_cand_pairs = pairs_adapt
    test_eng_u = test_indices[test_cand_pairs[:, 0]]
    test_eng_v = test_indices[test_cand_pairs[:, 1]]
    test_eng_pairs_fwd = np.column_stack([test_eng_u, test_eng_v])
    test_eng_pairs_rev = np.column_stack([test_eng_v, test_eng_u])

    X_test_cand_fwd = preprocessor.transform(engine.pair_matrix_vec(test_eng_pairs_fwd))
    X_test_cand_rev = preprocessor.transform(engine.pair_matrix_vec(test_eng_pairs_rev))

    p_same_test_raw, _ = same_model.predict_proba_symmetric(X_test_cand_fwd, X_test_cand_rev)
    p_next_test_raw, p_prev_test_raw, _, _ = next_model.predict_proba_directed(X_test_cand_fwd, X_test_cand_rev)

    test_edge_features = np.column_stack([
        p_same_test_raw,
        p_next_test_raw,
        p_prev_test_raw,
        X_test_cand_fwd[:, 0],
        X_test_cand_fwd[:, 1],
        np.zeros(len(test_cand_pairs)),
    ]).astype(np.float32)

    y_next_test = np.array([
        1 if test_obs_fragments[u].fragment_id in test_gt_successors and
             test_gt_successors[test_obs_fragments[u].fragment_id] == test_obs_fragments[v].fragment_id else 0
        for u, v in test_cand_pairs
    ], dtype=np.float32)

    y_same_test = np.array([
        1 if test_gt_fragments[u].source_file_id == test_gt_fragments[v].source_file_id else 0
        for u, v in test_cand_pairs
    ], dtype=np.float32)

    # Train GraphSAGE
    logger.info("Fitting GraphSAGERefiner...")
    sage_refiner = GraphSAGERefiner(node_dim=train_single_features.shape[1], hidden_dim=64, edge_dim=6, epochs=30)
    sage_refiner.fit(train_single_features, train_cand_pairs, train_edge_features, y_next_train, y_same_train)
    p_next_sage, p_same_sage = sage_refiner.predict_refined(test_single_features, test_cand_pairs, test_edge_features)

    # Train GAT
    logger.info("Fitting GATRefiner...")
    gat_refiner = GATRefiner(node_dim=train_single_features.shape[1], hidden_dim=64, edge_dim=6, epochs=30)
    gat_refiner.fit(train_single_features, train_cand_pairs, train_edge_features, y_next_train, y_same_train)
    p_next_gat, p_same_gat = gat_refiner.predict_refined(test_single_features, test_cand_pairs, test_edge_features)

    def compute_edge_metrics(y_true: np.ndarray, y_score: np.ndarray, name: str) -> Dict[str, float]:
        auc = roc_auc_score(y_true, y_score) if np.sum(y_true) > 0 else 0.5
        pr_auc = average_precision_score(y_true, y_score) if np.sum(y_true) > 0 else 0.0
        return {
            "model": name,
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
        }

    gnn_results = {
        "pairwise_baseline": {
            "next_fragment": compute_edge_metrics(y_next_test, p_next_test_raw, "pairwise_baseline"),
            "same_file": compute_edge_metrics(y_same_test, p_same_test_raw, "pairwise_baseline"),
        },
        "graphsage_refined": {
            "next_fragment": compute_edge_metrics(y_next_test, p_next_sage, "graphsage"),
            "same_file": compute_edge_metrics(y_same_test, p_same_sage, "graphsage"),
        },
        "gat_refined": {
            "next_fragment": compute_edge_metrics(y_next_test, p_next_gat, "gat"),
            "same_file": compute_edge_metrics(y_same_test, p_same_gat, "gat"),
        },
    }
    logger.info("Edge Evaluation: Baseline Next ROC=%.4f PR=%.4f | GraphSAGE Next ROC=%.4f PR=%.4f | GAT Next ROC=%.4f PR=%.4f",
                gnn_results["pairwise_baseline"]["next_fragment"]["roc_auc"],
                gnn_results["pairwise_baseline"]["next_fragment"]["pr_auc"],
                gnn_results["graphsage_refined"]["next_fragment"]["roc_auc"],
                gnn_results["graphsage_refined"]["next_fragment"]["pr_auc"],
                gnn_results["gat_refined"]["next_fragment"]["roc_auc"],
                gnn_results["gat_refined"]["next_fragment"]["pr_auc"])

    # -------------------------------------------------------------
    # 4. Multi-Hypothesis Beam Search Reconstruction
    # -------------------------------------------------------------
    logger.info(">>> 4. Benchmarking Multi-Hypothesis Beam Search (W=1, 3, 5, 10)...")
    eval_engine = EvaluationEngine()
    test_pool_map = {f.observable.fragment_id: f.original_bytes for f in test_gt_fragments}
    gt_manifest = {"files": file_metadata}
    test_pool_id_set = set(test_pool_map.keys())

    # Build refined graph with GraphSAGE probabilities
    g_final = nx.DiGraph()
    frag_ids = [f.fragment_id for f in test_obs_fragments]
    for fid in frag_ids:
        g_final.add_node(fid)

    for idx, (u, v) in enumerate(test_cand_pairs):
        u_id = frag_ids[u]
        v_id = frag_ids[v]
        g_final.add_edge(
            u_id,
            v_id,
            p_next=float(p_next_sage[idx]),
            p_same=float(p_same_sage[idx]),
            boundary_compat=float(test_edge_features[idx, 4]),
        )

    # Generate same-file clusters using refined P_same
    ug = nx.Graph()
    ug.add_nodes_from(frag_ids)
    for u, v, d in g_final.edges(data=True):
        if d.get("p_same", 0.0) >= 0.25:
            ug.add_edge(u, v)
    candidate_clusters = [list(c) for c in nx.connected_components(ug)]
    candidate_clusters.sort(key=lambda c: len(c), reverse=True)

    reconstruction_results = {}
    for w_val in [1, 3, 5, 10]:
        t0_recon = time.time()
        bs_engine = MultiHypothesisBeamSearchV2(beam_width=w_val, max_hypotheses=3, tau_next_min=0.05)
        recons, _ = bs_engine.reconstruct_all(g_final, test_pool_map, candidate_clusters=candidate_clusters)
        dur = time.time() - t0_recon

        eff = {
            "runtime_seconds": round(dur, 4),
            "candidate_graph_nodes": g_final.number_of_nodes(),
            "candidate_graph_edges": g_final.number_of_edges(),
        }
        res_w = eval_engine.evaluate_reconstruction(
            gt_manifest, recons, all_pool_fragment_ids=test_pool_id_set, efficiency_stats=eff
        )
        reconstruction_results[f"beam_width_{w_val}"] = {
            "beam_width": w_val,
            "metrics": res_w,
        }
        logger.info("Beam Width W=%d: Reconstructed Files=%d, Merge Rate=%.4f, Partial=%.4f, Runtime=%.3fs",
                    w_val, len(recons), res_w["file_level"]["false_file_merge_rate"],
                    res_w["file_level"]["partial_reconstruction_rate"], dur)

    total_time = time.time() - start_time

    # -------------------------------------------------------------
    # 5. Compile Final JSON
    # -------------------------------------------------------------
    results = {
        "metadata": {
            "version": "2.0.0-phase4",
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(total_time, 2),
            "n_test_fragments": len(test_obs_fragments),
            "n_test_files": len(test_files),
        },
        "candidate_generation_benchmark": cand_results,
        "consistency_audit": consistency_results,
        "gnn_refinement_benchmark": gnn_results,
        "beam_search_reconstruction_benchmark": reconstruction_results,
    }

    results_path = base_dir / "phase4_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Phase 4 benchmark successfully written to phase4_results.json!")
    return results


if __name__ == "__main__":
    run_phase4_benchmark()
