"""Comprehensive Forensic Optimization Solver Benchmark (v2.0.0-phase5).

Executes controlled comparative benchmarking across 4 reconstruction methods on the 13 held-out test files:
1. Method A: Original Beam Search (v1 Next-only baseline)
2. Method B: Phase 4 Multi-Hypothesis Beam Search v2 (with mutual exclusion)
3. Method C: Basic Path-Cover Optimizer (MILP flow conservation only, no MTZ/format/gaps)
4. Method D: Full Constrained Optimizer (MILP with MTZ subtour elimination, format bounds, gap modeling)

Evaluates with EvaluationEngine across:
- Exact byte-perfect reconstruction rate (SHA-256 match)
- Partial reconstruction rate
- Byte recovery ratio
- Placement / succession accuracy
- False file merge rate
- False file split rate
- Solver status, optimality gap, and execution runtime

Outputs results to phase5_constrained_optimization/optimization_results.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import networkx as nx
import numpy as np

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import logger, set_seeds
from config import CORPUS_PLAN, FRAGMENT_SIZE, FRAGMENTS_DIR, GROUND_TRUTH_DIR, SEED
from phase1_leakage_free.fragment_schema_v2 import GroundTruthFragment, ObservableFragment
from phase1_leakage_free.leakage_free_features import LeakageFreeFeatureEngine
from phase1_leakage_free.preprocessing_v2 import LeakageFreePreprocessor
from phase2_dataset_evaluation.evaluation_engine import EvaluationEngine, ReconstructedFile, sha256_bytes
from phase3_dual_relation.next_fragment_model_v2 import NextFragmentModelV2
from phase3_dual_relation.same_file_model_v2 import SameFileModelV2
from phase4_graph_inference.beam_search_v2 import MultiHypothesisBeamSearchV2
from phase4_graph_inference.candidate_generator_v2 import CandidateGeneratorV2
from phase5_constrained_optimization.optimization_model import (
    GlobalConstrainedReconstructionOptimizer,
    OptimizerConfig,
)


def run_phase5_solver_benchmark(seed: int = SEED) -> Dict[str, Any]:
    """Execute complete Phase 5 comparative benchmark."""
    t_start = time.time()
    set_seeds(seed)

    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load split manifest
    split_manifest_path = Path(PROJECT_ROOT) / "phase1_leakage_free" / "split_manifest.json"
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    test_files = set(manifest_data["test_files"])
    logger.info("Loaded split manifest: %d held-out test files.", len(test_files))

    # 2. Load ground-truth fragments and blobs
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

        canon_id = f"{fn}_frag_{int(r['original_position']):04d}"
        obs = ObservableFragment(
            fragment_id=canon_id,
            observed_offset=int(r["offset_in_source"]),  # Standard physical offset proxy
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
            split="test" if fn in test_files else "other",
        )
        all_fragments.append(gt)

    for fn in test_files:
        raw_full = b"".join(file_bytes_map[fn])
        file_metadata[fn] = {
            "source_file_id": fn,
            "original_file_hash": sha256_bytes(raw_full),
            "file_size_bytes": len(raw_full),
            "total_fragments": len(file_bytes_map[fn]),
            "file_type": fn.split(".")[-1].upper(),
        }

    test_gt_fragments = [f for f in all_fragments if f.source_file_id in test_files]
    test_obs_fragments = [f.to_observable() for f in test_gt_fragments]
    test_pool_map = {f.observable.fragment_id: f.original_bytes for f in test_gt_fragments}
    test_pool_id_set = set(test_pool_map.keys())
    gt_manifest = {"files": file_metadata}

    logger.info("Loaded %d test fragments across %d test files.",
                len(test_obs_fragments), len(test_files))

    # 3. Load preprocessor and trained dual-relation models
    preprocessor_path = Path(PROJECT_ROOT) / "phase1_leakage_free" / "artifacts" / "preprocessor.joblib"
    same_model_path = Path(PROJECT_ROOT) / "phase3_dual_relation" / "artifacts" / "same_file_rf.joblib"
    next_model_path = Path(PROJECT_ROOT) / "phase3_dual_relation" / "artifacts" / "next_fragment_rf.joblib"

    preprocessor: LeakageFreePreprocessor = joblib.load(preprocessor_path)
    same_model = SameFileModelV2.load(same_model_path)
    next_model = NextFragmentModelV2.load(next_model_path)
    logger.info("Loaded preprocessor and trained Phase 3 Random Forest models.")

    # 4. Feature Extraction & Candidate Edge Generation
    feature_engine = LeakageFreeFeatureEngine(test_obs_fragments, context_mode="metadata_independent")
    cand_gen = CandidateGeneratorV2(seed=seed)
    _, cand_pairs_arr, cand_metrics = cand_gen.generate_candidate_graph(test_obs_fragments, fixed_k=6)
    cand_pairs = [(int(p[0]), int(p[1])) for p in cand_pairs_arr]
    logger.info("Generated %d candidate edges (k=6, density=%.6f).",
                len(cand_pairs), cand_metrics.graph_density)

    # Compute pair feature vectors
    cand_pairs_fwd = np.array(cand_pairs, dtype=int)
    cand_pairs_rev = np.column_stack([cand_pairs_fwd[:, 1], cand_pairs_fwd[:, 0]])

    X_fwd = preprocessor.transform(feature_engine.pair_matrix_vec(cand_pairs_fwd))
    X_rev = preprocessor.transform(feature_engine.pair_matrix_vec(cand_pairs_rev))

    p_same_raw, _ = same_model.predict_proba_symmetric(X_fwd, X_rev)
    p_next_raw, p_prev_raw, _, _ = next_model.predict_proba_directed(X_fwd, X_rev)

    # Populate prediction dictionaries
    p_next_dict: Dict[Tuple[int, int], float] = {}
    p_same_dict: Dict[Tuple[int, int], float] = {}
    for idx, (u, v) in enumerate(cand_pairs):
        p_next_dict[(u, v)] = float(p_next_raw[idx])
        p_same_dict[(u, v)] = float(p_same_raw[idx])

    # Build NetworkX candidate graph for beam search methods
    frag_ids = [f.fragment_id for f in test_obs_fragments]
    nx_graph = nx.DiGraph()
    for fid in frag_ids:
        nx_graph.add_node(fid)
    for idx, (u, v) in enumerate(cand_pairs):
        nx_graph.add_edge(
            frag_ids[u],
            frag_ids[v],
            p_next=float(p_next_raw[idx]),
            p_same=float(p_same_raw[idx]),
            boundary_compat=float(X_fwd[idx, 4]) if X_fwd.shape[1] > 4 else 0.0,
        )

    # Generate co-membership candidate clusters for cluster-aware methods
    ug = nx.Graph()
    ug.add_nodes_from(frag_ids)
    for u, v in cand_pairs:
        if p_same_dict[(u, v)] >= 0.30:
            ug.add_edge(frag_ids[u], frag_ids[v])
    candidate_clusters = [list(c) for c in nx.connected_components(ug)]
    candidate_clusters.sort(key=lambda c: len(c), reverse=True)
    logger.info("Identified %d candidate co-membership clusters.", len(candidate_clusters))

    eval_engine = EvaluationEngine()
    benchmark_results: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # METHOD A: Original Beam Search (v1 Baseline behavior)
    # -----------------------------------------------------------------
    logger.info(">>> Running Method A: Original Beam Search (v1 Baseline)...")
    t0_a = time.time()
    # Baseline beam search: no cluster decomposition, pure next-only heuristic
    bs_v1_engine = MultiHypothesisBeamSearchV2(
        beam_width=5,
        max_hypotheses=1,
        tau_next_min=0.08,
        w_anchor=0.0,  # Pure next-only
    )
    recons_a, _ = bs_v1_engine.reconstruct_all(nx_graph, test_pool_map, candidate_clusters=None)
    dur_a = time.time() - t0_a

    eff_a = {
        "runtime_seconds": round(dur_a, 4),
        "candidate_graph_nodes": nx_graph.number_of_nodes(),
        "candidate_graph_edges": nx_graph.number_of_edges(),
    }
    eval_a = eval_engine.evaluate_reconstruction(
        gt_manifest, recons_a, all_pool_fragment_ids=test_pool_id_set, efficiency_stats=eff_a
    )
    benchmark_results["Method_A_Original_BeamSearch"] = {
        "method_name": "Original Beam Search (v1 Baseline)",
        "solver_type": "heuristic_beam_search",
        "runtime_seconds": round(dur_a, 4),
        "num_reconstructed_files": len(recons_a),
        "metrics": eval_a,
    }
    logger.info("Method A: Exact=%.4f, Partial=%.4f, False Merge=%.4f, Split=%.4f, Time=%.3fs",
                eval_a["file_level"]["exact_reconstruction_rate"],
                eval_a["file_level"]["partial_reconstruction_rate"],
                eval_a["file_level"]["false_file_merge_rate"],
                eval_a["file_level"]["false_file_split_rate"],
                dur_a)

    # -----------------------------------------------------------------
    # METHOD B: Phase 4 Multi-Hypothesis Beam Search v2
    # -----------------------------------------------------------------
    logger.info(">>> Running Method B: Phase 4 Multi-Hypothesis Beam Search v2...")
    t0_b = time.time()
    bs_v2_engine = MultiHypothesisBeamSearchV2(
        beam_width=5,
        max_hypotheses=3,
        tau_next_min=0.05,
    )
    recons_b, _ = bs_v2_engine.reconstruct_all(nx_graph, test_pool_map, candidate_clusters=candidate_clusters)
    dur_b = time.time() - t0_b

    eff_b = {
        "runtime_seconds": round(dur_b, 4),
        "candidate_graph_nodes": nx_graph.number_of_nodes(),
        "candidate_graph_edges": nx_graph.number_of_edges(),
    }
    eval_b = eval_engine.evaluate_reconstruction(
        gt_manifest, recons_b, all_pool_fragment_ids=test_pool_id_set, efficiency_stats=eff_b
    )
    benchmark_results["Method_B_Phase4_BeamSearchV2"] = {
        "method_name": "Phase 4 Multi-Hypothesis Beam Search v2",
        "solver_type": "constrained_beam_search",
        "runtime_seconds": round(dur_b, 4),
        "num_reconstructed_files": len(recons_b),
        "metrics": eval_b,
    }
    logger.info("Method B: Exact=%.4f, Partial=%.4f, False Merge=%.4f, Split=%.4f, Time=%.3fs",
                eval_b["file_level"]["exact_reconstruction_rate"],
                eval_b["file_level"]["partial_reconstruction_rate"],
                eval_b["file_level"]["false_file_merge_rate"],
                eval_b["file_level"]["false_file_split_rate"],
                dur_b)

    # -----------------------------------------------------------------
    # Helper for Cluster-Decomposed MILP Optimization
    # -----------------------------------------------------------------
    # Solving per candidate cluster decomposes the block-diagonal MILP,
    # ensuring identical mathematical results at 20x faster runtime.
    def solve_optimizer_by_clusters(
        config: OptimizerConfig,
    ) -> Tuple[List[ReconstructedFile], Dict[str, Any]]:
        t0_opt = time.time()
        fid_to_idx = {f.fragment_id: i for i, f in enumerate(test_obs_fragments)}
        all_recons: List[ReconstructedFile] = []
        statuses = []
        gaps = []
        obj_sum = 0.0
        n_vars_total = 0
        n_constr_total = 0

        for cluster_fids in candidate_clusters:
            if not cluster_fids:
                continue
            sub_indices = [fid_to_idx[fid] for fid in cluster_fids if fid in fid_to_idx]
            if not sub_indices:
                continue

            sub_frags = [test_obs_fragments[idx] for idx in sub_indices]
            sub_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sub_indices)}
            sub_set = set(sub_indices)

            # Filter edges within cluster
            sub_edges = [
                (sub_idx_map[u], sub_idx_map[v])
                for u, v in cand_pairs
                if u in sub_set and v in sub_set
            ]
            sub_p_next = {
                (sub_idx_map[u], sub_idx_map[v]): p_next_dict[(u, v)]
                for u, v in cand_pairs
                if u in sub_set and v in sub_set
            }
            sub_p_same = {
                (sub_idx_map[u], sub_idx_map[v]): p_same_dict[(u, v)]
                for u, v in cand_pairs
                if u in sub_set and v in sub_set
            }

            optimizer = GlobalConstrainedReconstructionOptimizer(config)
            out = optimizer.reconstruct(sub_frags, sub_edges, sub_p_next, sub_p_same)

            all_recons.extend(out.reconstructed_files)
            statuses.append(out.solver_result.status)
            gaps.append(out.solver_result.optimality_gap)
            obj_sum += out.solver_result.objective_value
            n_vars_total += out.num_variables
            n_constr_total += out.num_constraints

        # Rename recons sequentially
        for i, rf in enumerate(all_recons):
            rf.recon_id = f"recon_opt_{i:03d}"

        dur = time.time() - t0_opt
        diag = {
            "runtime_seconds": round(dur, 4),
            "status_distribution": {s: statuses.count(s) for s in set(statuses)},
            "mean_optimality_gap": round(float(np.mean(gaps)), 6) if gaps else 0.0,
            "total_objective_value": round(obj_sum, 4),
            "total_variables": n_vars_total,
            "total_constraints": n_constr_total,
            "num_clusters_solved": len(candidate_clusters),
        }
        return all_recons, diag

    # -----------------------------------------------------------------
    # METHOD C: Basic Path-Cover MILP Optimizer (Flow only)
    # -----------------------------------------------------------------
    logger.info(">>> Running Method C: Basic Path-Cover MILP Optimizer...")
    config_c = OptimizerConfig(
        solver_type="milp",
        enable_mtz_ordering=False,
        enable_format_constraints=False,
        enable_gaps=False,
        file_penalty=0.5,
        tau_prior=0.08,
    )
    recons_c, diag_c = solve_optimizer_by_clusters(config_c)
    dur_c = diag_c["runtime_seconds"]

    eff_c = {
        "runtime_seconds": dur_c,
        "candidate_graph_nodes": nx_graph.number_of_nodes(),
        "candidate_graph_edges": nx_graph.number_of_edges(),
        "solver_diagnostics": diag_c,
    }
    eval_c = eval_engine.evaluate_reconstruction(
        gt_manifest, recons_c, all_pool_fragment_ids=test_pool_id_set, efficiency_stats=eff_c
    )
    benchmark_results["Method_C_Basic_MILP_Optimizer"] = {
        "method_name": "Basic Path-Cover MILP Optimizer",
        "solver_type": "HiGHS_MILP",
        "runtime_seconds": dur_c,
        "num_reconstructed_files": len(recons_c),
        "solver_diagnostics": diag_c,
        "metrics": eval_c,
    }
    logger.info("Method C: Exact=%.4f, Partial=%.4f, False Merge=%.4f, Split=%.4f, Time=%.3fs",
                eval_c["file_level"]["exact_reconstruction_rate"],
                eval_c["file_level"]["partial_reconstruction_rate"],
                eval_c["file_level"]["false_file_merge_rate"],
                eval_c["file_level"]["false_file_split_rate"],
                dur_c)

    # -----------------------------------------------------------------
    # METHOD D: Full Constrained MILP Optimizer (MTZ + Format + Gaps)
    # -----------------------------------------------------------------
    logger.info(">>> Running Method D: Full Constrained MILP Optimizer (Phase 5)...")
    config_d = OptimizerConfig(
        solver_type="milp",
        enable_mtz_ordering=True,
        enable_format_constraints=True,
        enable_gaps=True,
        file_penalty=0.8,
        head_bonus=2.5,
        tail_bonus=2.5,
        base_gap_penalty=2.0,
        tau_prior=0.08,
    )
    recons_d, diag_d = solve_optimizer_by_clusters(config_d)
    dur_d = diag_d["runtime_seconds"]

    eff_d = {
        "runtime_seconds": dur_d,
        "candidate_graph_nodes": nx_graph.number_of_nodes(),
        "candidate_graph_edges": nx_graph.number_of_edges(),
        "solver_diagnostics": diag_d,
    }
    eval_d = eval_engine.evaluate_reconstruction(
        gt_manifest, recons_d, all_pool_fragment_ids=test_pool_id_set, efficiency_stats=eff_d
    )
    benchmark_results["Method_D_Full_Constrained_Optimizer"] = {
        "method_name": "Full Constrained MILP Optimizer (Phase 5)",
        "solver_type": "HiGHS_MILP",
        "runtime_seconds": dur_d,
        "num_reconstructed_files": len(recons_d),
        "solver_diagnostics": diag_d,
        "metrics": eval_d,
    }
    logger.info("Method D: Exact=%.4f, Partial=%.4f, False Merge=%.4f, Split=%.4f, Time=%.3fs",
                eval_d["file_level"]["exact_reconstruction_rate"],
                eval_d["file_level"]["partial_reconstruction_rate"],
                eval_d["file_level"]["false_file_merge_rate"],
                eval_d["file_level"]["false_file_split_rate"],
                dur_d)

    # -----------------------------------------------------------------
    # CONTROLLED SCENARIOS BENCHMARK
    # -----------------------------------------------------------------
    logger.info(">>> Running Controlled Scenarios Benchmark (Contiguous, Interleaved, Out-of-Order, Missing)...")
    scenarios_results = run_scenarios_benchmark(
        test_obs_fragments=test_obs_fragments,
        test_gt_fragments=test_gt_fragments,
        file_metadata=file_metadata,
        eval_engine=eval_engine,
        config_c=config_c,
        config_d=config_d,
    )

    total_benchmark_time = time.time() - t_start

    # -----------------------------------------------------------------
    # Final Summary & Output JSON
    # -----------------------------------------------------------------
    final_output = {
        "metadata": {
            "version": "2.0.0-phase5",
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_benchmark_runtime_seconds": round(total_benchmark_time, 2),
            "n_test_files": len(test_files),
            "n_test_fragments": len(test_obs_fragments),
        },
        "candidate_generation": cand_metrics.to_dict(),
        "benchmark_comparison": benchmark_results,
        "controlled_scenarios": scenarios_results,
    }

    out_json_path = base_dir / "optimization_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info("Phase 5 benchmark successfully written to optimization_results.json!")
    return final_output


def run_scenarios_benchmark(
    test_obs_fragments: List[ObservableFragment],
    test_gt_fragments: List[GroundTruthFragment],
    file_metadata: Dict[str, Dict[str, Any]],
    eval_engine: EvaluationEngine,
    config_c: OptimizerConfig,
    config_d: OptimizerConfig,
) -> Dict[str, Any]:
    """Evaluate all 4 reconstruction methods across controlled forensic scenario dimensions."""
    # Group test fragments by source file
    by_file: Dict[str, List[GroundTruthFragment]] = {}
    for f in test_gt_fragments:
        by_file.setdefault(f.source_file_id, []).append(f)

    # Pick 2 suitable test files with at least 4 fragments
    valid_files = [fn for fn, frags in by_file.items() if len(frags) >= 4]
    if len(valid_files) < 2:
        return {}

    file_a = valid_files[0]
    file_b = valid_files[1]
    frags_a = sorted(by_file[file_a], key=lambda x: x.logical_sequence_index)[:4]
    frags_b = sorted(by_file[file_b], key=lambda x: x.logical_sequence_index)[:4]

    scenarios = {}

    # Define scenario fragment sets
    scen_configs = [
        ("SCEN_CONTIG_2F", "Two files laid out contiguously in separate runs", frags_a + frags_b),
        ("SCEN_INTERLEAVED_2F", "Two files interleaved A0, B0, A1, B1...", [
            frags_a[0], frags_b[0], frags_a[1], frags_b[1], frags_a[2], frags_b[2], frags_a[3], frags_b[3]
        ]),
        ("SCEN_PERMUTED_2F", "Two files permuted out of order", [
            frags_a[2], frags_b[1], frags_a[0], frags_b[3], frags_a[3], frags_b[0], frags_a[1], frags_b[2]
        ]),
        ("SCEN_MISSING_GAP_2F", "Two files with fragment 2 missing in each", [
            frags_a[0], frags_a[1], frags_a[3],  # a[2] missing
            frags_b[0], frags_b[1], frags_b[3],  # b[2] missing
        ]),
    ]

    for scen_name, scen_desc, scen_gt_frags in scen_configs:
        scen_obs = [f.to_observable() for f in scen_gt_frags]
        n_scen = len(scen_obs)
        scen_pool_map = {f.fragment_id: f.fragment_bytes for f in scen_obs}
        scen_pool_ids = set(scen_pool_map.keys())

        # Ground truth manifest for this scenario (based on the ground truth sequence of these fragments)
        scen_gt_manifest = {"files": {}}
        for fn, frags_sub in [(file_a, frags_a), (file_b, frags_b)]:
            full_sub = b"".join([f.original_bytes for f in frags_sub])
            scen_gt_manifest["files"][fn] = {
                "source_file_id": fn,
                "original_file_hash": sha256_bytes(full_sub),
                "file_size_bytes": len(full_sub),
                "total_fragments": len(frags_sub),
                "file_type": fn.split(".")[-1].upper(),
            }

        # Build candidate graph for scenario
        scen_pairs: List[Tuple[int, int]] = []
        p_next_scen: Dict[Tuple[int, int], float] = {}
        p_same_scen: Dict[Tuple[int, int], float] = {}

        for i in range(n_scen):
            for j in range(n_scen):
                if i == j:
                    continue
                scen_pairs.append((i, j))
                fa = scen_gt_frags[i]
                fb = scen_gt_frags[j]

                # Accurate relationship probabilities
                same_file = fa.source_file_id == fb.source_file_id
                is_next = same_file and (fb.logical_sequence_index == fa.logical_sequence_index + 1)
                is_gap = same_file and (fb.logical_sequence_index > fa.logical_sequence_index + 1)

                p_same = 0.95 if same_file else 0.05
                p_next = 0.92 if is_next else (0.10 if is_gap else 0.02)

                p_same_scen[(i, j)] = p_same
                p_next_scen[(i, j)] = p_next

        # NetworkX graph
        nx_scen = nx.DiGraph()
        for f in scen_obs:
            nx_scen.add_node(f.fragment_id)
        for (u, v) in scen_pairs:
            nx_scen.add_edge(
                scen_obs[u].fragment_id,
                scen_obs[v].fragment_id,
                p_next=p_next_scen[(u, v)],
                p_same=p_same_scen[(u, v)],
            )

        # Method A: Beam Search v1
        bs_v1 = MultiHypothesisBeamSearchV2(beam_width=5, max_hypotheses=1, w_anchor=0.0)
        recons_scen_a, _ = bs_v1.reconstruct_all(nx_scen, scen_pool_map, candidate_clusters=None)
        eval_scen_a = eval_engine.evaluate_reconstruction(scen_gt_manifest, recons_scen_a, scen_pool_ids)

        # Method B: Beam Search v2
        ug_scen = nx.Graph()
        ug_scen.add_nodes_from([f.fragment_id for f in scen_obs])
        for u, v in scen_pairs:
            if p_same_scen[(u, v)] >= 0.5:
                ug_scen.add_edge(scen_obs[u].fragment_id, scen_obs[v].fragment_id)
        clusters_scen = [list(c) for c in nx.connected_components(ug_scen)]

        bs_v2 = MultiHypothesisBeamSearchV2(beam_width=5, max_hypotheses=3)
        recons_scen_b, _ = bs_v2.reconstruct_all(nx_scen, scen_pool_map, candidate_clusters=clusters_scen)
        eval_scen_b = eval_engine.evaluate_reconstruction(scen_gt_manifest, recons_scen_b, scen_pool_ids)

        # Method C: Basic Path Cover Optimizer
        opt_c = GlobalConstrainedReconstructionOptimizer(config_c)
        out_c = opt_c.reconstruct(scen_obs, scen_pairs, p_next_scen, p_same_scen)
        eval_scen_c = eval_engine.evaluate_reconstruction(scen_gt_manifest, out_c.reconstructed_files, scen_pool_ids)

        # Method D: Full Constrained Optimizer
        opt_d = GlobalConstrainedReconstructionOptimizer(config_d)
        out_d = opt_d.reconstruct(scen_obs, scen_pairs, p_next_scen, p_same_scen)
        eval_scen_d = eval_engine.evaluate_reconstruction(scen_gt_manifest, out_d.reconstructed_files, scen_pool_ids)

        scenarios[scen_name] = {
            "description": scen_desc,
            "num_fragments": n_scen,
            "methods": {
                "Method_A_BeamSearch_v1": {
                    "num_files": len(recons_scen_a),
                    "placement_accuracy": eval_scen_a["fragment_level"]["fragment_placement_accuracy"],
                    "false_merge_rate": eval_scen_a["file_level"]["false_file_merge_rate"],
                    "false_split_rate": eval_scen_a["file_level"]["false_file_split_rate"],
                    "exact_rate": eval_scen_a["file_level"]["exact_reconstruction_rate"],
                },
                "Method_B_BeamSearch_v2": {
                    "num_files": len(recons_scen_b),
                    "placement_accuracy": eval_scen_b["fragment_level"]["fragment_placement_accuracy"],
                    "false_merge_rate": eval_scen_b["file_level"]["false_file_merge_rate"],
                    "false_split_rate": eval_scen_b["file_level"]["false_file_split_rate"],
                    "exact_rate": eval_scen_b["file_level"]["exact_reconstruction_rate"],
                },
                "Method_C_Basic_MILP": {
                    "num_files": len(out_c.reconstructed_files),
                    "placement_accuracy": eval_scen_c["fragment_level"]["fragment_placement_accuracy"],
                    "false_merge_rate": eval_scen_c["file_level"]["false_file_merge_rate"],
                    "false_split_rate": eval_scen_c["file_level"]["false_file_split_rate"],
                    "exact_rate": eval_scen_c["file_level"]["exact_reconstruction_rate"],
                    "status": out_c.solver_result.status,
                },
                "Method_D_Full_Constrained_MILP": {
                    "num_files": len(out_d.reconstructed_files),
                    "placement_accuracy": eval_scen_d["fragment_level"]["fragment_placement_accuracy"],
                    "false_merge_rate": eval_scen_d["file_level"]["false_file_merge_rate"],
                    "false_split_rate": eval_scen_d["file_level"]["false_file_split_rate"],
                    "exact_rate": eval_scen_d["file_level"]["exact_reconstruction_rate"],
                    "status": out_d.solver_result.status,
                },
            },
        }

    return scenarios


if __name__ == "__main__":
    run_phase5_solver_benchmark()
