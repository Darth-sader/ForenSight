"""Forensic Reconstruction Evaluation Engine (v2.0.0-phase2).

Implements rigorous fragment-level, file-level, robustness, and efficiency metrics
for evaluating forensic multi-file reconstruction under corruption, interleaving,
and missing fragments.

All metric formulas define explicit mathematical numerators and denominators.
Partial reconstruction is strictly distinguished from exact byte-perfect recovery.
"""
from __future__ import annotations

import difflib
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass
class ReconstructedFile:
    """A file reconstructed by an inference / carving algorithm."""
    recon_id: str
    fragment_sequence: List[str]    # Ordered list of fragment_ids
    reconstructed_bytes: bytes      # Concatenated byte payload
    claimed_type: Optional[str] = None

    @property
    def file_hash(self) -> str:
        return sha256_bytes(self.reconstructed_bytes)


class EvaluationEngine:
    """Computes forensic fragment-level and file-level reconstruction benchmarks."""

    def __init__(self):
        pass

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute the length of the Longest Common Subsequence between two fragment lists."""
        m, n = len(seq1), len(seq2)
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if seq1[i - 1] == seq2[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    def evaluate_reconstruction(
        self,
        ground_truth_manifest: Dict[str, Any],
        reconstructed_files: List[ReconstructedFile],
        all_pool_fragment_ids: Optional[Set[str]] = None,
        efficiency_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute complete evaluation metrics against ground truth.

        Args:
            ground_truth_manifest: Dictionary adhering to ground_truth_schema_v2.json.
            reconstructed_files: List of ReconstructedFile objects emitted by carver.
            all_pool_fragment_ids: Optional set of all fragment_ids available in pool.
            efficiency_stats: Optional runtime, memory, graph size dictionary.
        """
        gt_files = ground_truth_manifest.get("files", {})
        total_gt_files = len(gt_files)
        total_recon_files = len(reconstructed_files)

        if total_gt_files == 0:
            raise ValueError("Ground-truth manifest contains 0 files!")

        # Map each fragment_id to its ground-truth source file and logical position
        frag_to_gt: Dict[str, Tuple[str, int]] = {}
        gt_file_sequences: Dict[str, List[str]] = {}
        gt_file_hashes: Dict[str, str] = {}
        gt_file_sizes: Dict[str, int] = {}
        gt_file_types: Dict[str, str] = {}
        total_gt_fragments = 0
        total_gt_bytes = 0

        for fn, fmeta in gt_files.items():
            gt_file_hashes[fn] = fmeta["original_file_hash"]
            gt_file_sizes[fn] = fmeta["file_size_bytes"]
            gt_file_types[fn] = fmeta.get("file_type", fn.split(".")[-1].upper())
            total_gt_bytes += fmeta["file_size_bytes"]

            # Ground truth logical sequence of fragment hashes / ids
            total_frags = fmeta["total_fragments"]
            total_gt_fragments += total_frags
            seq = [f"{fn}_frag_{i:04d}" for i in range(total_frags)]
            gt_file_sequences[fn] = seq
            for i, fid in enumerate(seq):
                frag_to_gt[fid] = (fn, i)

        # -----------------------------------------------------------------
        # 1. Fragment-Level Metrics
        # -----------------------------------------------------------------
        assigned_fragment_ids: List[str] = []
        for rf in reconstructed_files:
            assigned_fragment_ids.extend(rf.fragment_sequence)

        total_assigned_fragments = len(assigned_fragment_ids)
        unique_assigned_fragments = set(assigned_fragment_ids)

        # Duplicate assignment count
        n_duplicates = total_assigned_fragments - len(unique_assigned_fragments)
        duplicate_assignment_rate = (
            float(n_duplicates / total_assigned_fragments) if total_assigned_fragments > 0 else 0.0
        )

        # Unused fragment rate
        if all_pool_fragment_ids is not None and len(all_pool_fragment_ids) > 0:
            unused_frags = all_pool_fragment_ids - unique_assigned_fragments
            unused_fragment_rate = float(len(unused_frags) / len(all_pool_fragment_ids))
        else:
            unused_fragment_rate = 0.0

        # Fragment placement accuracy & Successor precision/recall
        correct_placements = 0
        valid_successors = 0
        total_recon_transitions = 0
        total_gt_transitions = sum(max(0, len(s) - 1) for s in gt_file_sequences.values())

        for rf in reconstructed_files:
            seq = rf.fragment_sequence
            for idx, fid in enumerate(seq):
                if fid in frag_to_gt:
                    fn, true_pos = frag_to_gt[fid]
                    if idx == true_pos:
                        correct_placements += 1

            for a in range(len(seq) - 1):
                total_recon_transitions += 1
                fa, fb = seq[a], seq[a + 1]
                if fa in frag_to_gt and fb in frag_to_gt:
                    fn_a, pos_a = frag_to_gt[fa]
                    fn_b, pos_b = frag_to_gt[fb]
                    if fn_a == fn_b and pos_b == pos_a + 1:
                        valid_successors += 1

        fragment_placement_accuracy = (
            float(correct_placements / total_gt_fragments) if total_gt_fragments > 0 else 0.0
        )
        successor_precision = (
            float(valid_successors / total_recon_transitions) if total_recon_transitions > 0 else 0.0
        )
        successor_recall = (
            float(valid_successors / total_gt_transitions) if total_gt_transitions > 0 else 0.0
        )
        successor_f1 = (
            float(2 * successor_precision * successor_recall / (successor_precision + successor_recall))
            if (successor_precision + successor_recall) > 0 else 0.0
        )

        # -----------------------------------------------------------------
        # 2. File-Level Metrics
        # -----------------------------------------------------------------
        exact_reconstructed_files = 0
        partial_reconstructed_files = 0
        total_correct_bytes_recovered = 0
        false_merged_files = 0
        false_reconstructed_files = 0

        # Track which GT files were recovered
        gt_recovery_status: Dict[str, Dict[str, Any]] = {
            fn: {"exact": False, "partial": False, "best_lcs": 0, "recon_matches": []}
            for fn in gt_files
        }

        # Track file splits: count how many recon files contain fragments from each GT file
        gt_fragments_in_recon: Dict[str, Set[str]] = {fn: set() for fn in gt_files}

        for rf in reconstructed_files:
            seq = rf.fragment_sequence
            # Count source files present in this reconstructed file
            sources_in_rf = {frag_to_gt[fid][0] for fid in seq if fid in frag_to_gt}

            if len(sources_in_rf) == 0:
                false_reconstructed_files += 1
                continue
            elif len(sources_in_rf) > 1:
                false_merged_files += 1

            for sfn in sources_in_rf:
                gt_fragments_in_recon[sfn].add(rf.recon_id)

            # Check exact hash match
            is_exact = False
            for fn, true_hash in gt_file_hashes.items():
                if rf.file_hash == true_hash:
                    exact_reconstructed_files += 1
                    is_exact = True
                    gt_recovery_status[fn]["exact"] = True
                    gt_recovery_status[fn]["recon_matches"].append(rf.recon_id)
                    total_correct_bytes_recovered += gt_file_sizes[fn]
                    break

            # If not exact, check partial LCS recovery
            if not is_exact and len(sources_in_rf) == 1:
                sfn = list(sources_in_rf)[0]
                true_seq = gt_file_sequences[sfn]
                lcs = self._lcs_length(seq, true_seq)
                if lcs > gt_recovery_status[sfn]["best_lcs"]:
                    gt_recovery_status[sfn]["best_lcs"] = lcs
                if lcs >= 0.5 * len(true_seq):
                    partial_reconstructed_files += 1
                    gt_recovery_status[sfn]["partial"] = True
                    # Estimate partially recovered bytes
                    frac = lcs / len(true_seq)
                    total_correct_bytes_recovered += int(round(frac * gt_file_sizes[sfn]))

        # False file split rate: GT files split across multiple reconstructed files
        n_split_files = sum(1 for fn, rset in gt_fragments_in_recon.items() if len(rset) > 1)
        false_file_split_rate = float(n_split_files / total_gt_files) if total_gt_files > 0 else 0.0

        exact_reconstruction_rate = float(exact_reconstructed_files / total_gt_files)
        partial_reconstruction_rate = float(partial_reconstructed_files / total_gt_files)
        byte_recovery_ratio = (
            float(total_correct_bytes_recovered / total_gt_bytes) if total_gt_bytes > 0 else 0.0
        )
        false_file_merge_rate = (
            float(false_merged_files / total_recon_files) if total_recon_files > 0 else 0.0
        )
        false_reconstruction_rate = (
            float(false_reconstructed_files / total_recon_files) if total_recon_files > 0 else 0.0
        )

        # -----------------------------------------------------------------
        # 3. Format-Specific Breakdown
        # -----------------------------------------------------------------
        format_breakdown: Dict[str, Dict[str, Any]] = {}
        by_fmt: Dict[str, List[str]] = {}
        for fn, fmt in gt_file_types.items():
            by_fmt.setdefault(fmt, []).append(fn)

        for fmt, flist in by_fmt.items():
            exact_count = sum(1 for fn in flist if gt_recovery_status[fn]["exact"])
            part_count = sum(1 for fn in flist if gt_recovery_status[fn]["partial"])
            format_breakdown[fmt] = {
                "total_files": len(flist),
                "exact_reconstructed": exact_count,
                "partial_reconstructed": part_count,
                "exact_recovery_rate": round(exact_count / len(flist), 4),
            }

        # -----------------------------------------------------------------
        # 4. Efficiency Metrics
        # -----------------------------------------------------------------
        eff = efficiency_stats or {}
        efficiency_metrics = {
            "runtime_seconds": eff.get("runtime_seconds", 0.0),
            "peak_memory_mb": eff.get("peak_memory_mb", 0.0),
            "candidate_graph_nodes": eff.get("candidate_graph_nodes", 0),
            "candidate_graph_edges": eff.get("candidate_graph_edges", 0),
            "pairwise_evaluations_count": eff.get("pairwise_evaluations_count", 0),
            "search_time_seconds": eff.get("search_time_seconds", 0.0),
        }

        return {
            "summary": {
                "total_ground_truth_files": total_gt_files,
                "total_reconstructed_files": total_recon_files,
                "exact_reconstructed_files": exact_reconstructed_files,
                "partial_reconstructed_files": partial_reconstructed_files,
                "exact_reconstruction_rate": round(exact_reconstruction_rate, 4),
                "partial_reconstruction_rate": round(partial_reconstruction_rate, 4),
                "byte_recovery_ratio": round(byte_recovery_ratio, 4),
            },
            "fragment_level": {
                "total_gt_fragments": total_gt_fragments,
                "total_assigned_fragments": total_assigned_fragments,
                "fragment_placement_accuracy": round(fragment_placement_accuracy, 4),
                "successor_precision": round(successor_precision, 4),
                "successor_recall": round(successor_recall, 4),
                "successor_f1": round(successor_f1, 4),
                "duplicate_assignment_rate": round(duplicate_assignment_rate, 4),
                "unused_fragment_rate": round(unused_fragment_rate, 4),
            },
            "file_level": {
                "exact_reconstruction_rate": round(exact_reconstruction_rate, 4),
                "partial_reconstruction_rate": round(partial_reconstruction_rate, 4),
                "byte_recovery_ratio": round(byte_recovery_ratio, 4),
                "false_file_merge_rate": round(false_file_merge_rate, 4),
                "false_file_split_rate": round(false_file_split_rate, 4),
                "false_reconstruction_rate": round(false_reconstruction_rate, 4),
            },
            "format_breakdown": format_breakdown,
            "efficiency": efficiency_metrics,
        }
