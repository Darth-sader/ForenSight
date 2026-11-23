"""Unit and Scenario Test Suite for Global Constrained Reconstruction (v2.0.0-phase5).

Validates the 10 core optimization requirements:
1.  One contiguous file reconstruction
2.  Multiple contiguous files
3.  Multiple interleaved files
4.  Fragmented non-contiguous files
5.  Missing intermediate fragments (gap variable activation)
6.  Corrupted fragments
7.  Contradictory high-scoring edges (forking resolution)
8.  Directed cycle elimination (MTZ subtour constraints)
9.  Duplicate fragment usage prevention (degree <= 1)
10. Infeasible constraint handling (graceful failure reporting)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase1_leakage_free.fragment_schema_v2 import ObservableFragment
from phase5_constrained_optimization.milp_solver import MILPReconstructionSolver
from phase5_constrained_optimization.optimization_constraints import ConstraintMatrixBuilder
from phase5_constrained_optimization.optimization_model import (
    GlobalConstrainedReconstructionOptimizer,
    OptimizerConfig,
)
from phase5_constrained_optimization.optimization_variables import OptimizationVariableBuilder


def _make_fragment(
    frag_id: str,
    offset: int,
    payload: bytes,
    region: str = "data_area",
) -> ObservableFragment:
    return ObservableFragment(
        fragment_id=frag_id,
        observed_offset=offset,
        fragment_bytes=payload,
        fragment_length=len(payload),
        allocation_status="unallocated",
        filesystem_region=region,
    )


# -------------------------------------------------------------------------
# Test 1: One contiguous file reconstruction
# -------------------------------------------------------------------------
def test_01_single_contiguous_file():
    # 4 contiguous PNG fragments
    png_hdr = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 500
    chunk1 = b"IDAT" + b"\xaa" * 508
    chunk2 = b"IDAT" + b"\xbb" * 508
    png_ftr = b"\x00" * 500 + b"IEND\xaeB`\x82"

    frags = [
        _make_fragment("frag_0", 0, png_hdr),
        _make_fragment("frag_1", 512, chunk1),
        _make_fragment("frag_2", 1024, chunk2),
        _make_fragment("frag_3", 1536, png_ftr),
    ]

    edges = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
    p_next = {(0, 1): 0.95, (1, 2): 0.92, (2, 3): 0.98, (0, 2): 0.15, (1, 3): 0.10}
    p_same = {e: 0.99 for e in edges}

    optimizer = GlobalConstrainedReconstructionOptimizer(OptimizerConfig(file_penalty=1.0))
    output = optimizer.reconstruct(frags, edges, p_next, p_same)

    assert output.solver_result.status == "OPTIMAL"
    assert output.solver_result.success is True
    assert len(output.reconstructed_files) == 1
    assert output.reconstructed_files[0].fragment_sequence == ["frag_0", "frag_1", "frag_2", "frag_3"]


# -------------------------------------------------------------------------
# Test 2: Multiple contiguous files
# -------------------------------------------------------------------------
def test_02_multiple_contiguous_files():
    # File A: JPEG (frags 0, 1)
    jpg_hdr = b"\xff\xd8\xff\xe0" + b"\x00" * 508
    jpg_ftr = b"\x00" * 500 + b"\xff\xd9\x00\x00\x00\x00\x00\x00\x00\x00"
    # File B: PNG (frags 2, 3)
    png_hdr = b"\x89PNG\r\n\x1a\n" + b"\x00" * 504
    png_ftr = b"\x00" * 500 + b"IEND\xaeB`\x82"

    frags = [
        _make_fragment("jpg_0", 0, jpg_hdr),
        _make_fragment("jpg_1", 512, jpg_ftr),
        _make_fragment("png_0", 1024, png_hdr),
        _make_fragment("png_1", 1536, png_ftr),
    ]

    edges = [
        (0, 1), (2, 3),
        (0, 3), (2, 1),  # Cross-format distractor edges
    ]
    p_next = {(0, 1): 0.95, (2, 3): 0.95, (0, 3): 0.05, (2, 1): 0.05}
    p_same = {(0, 1): 0.98, (2, 3): 0.98, (0, 3): 0.02, (2, 1): 0.02}

    optimizer = GlobalConstrainedReconstructionOptimizer(OptimizerConfig(file_penalty=0.5))
    output = optimizer.reconstruct(frags, edges, p_next, p_same)

    assert output.solver_result.status == "OPTIMAL"
    assert len(output.reconstructed_files) == 2
    seqs = [rf.fragment_sequence for rf in output.reconstructed_files]
    assert ["jpg_0", "jpg_1"] in seqs
    assert ["png_0", "png_1"] in seqs


# -------------------------------------------------------------------------
# Test 3: Multiple interleaved files
# -------------------------------------------------------------------------
def test_03_multiple_interleaved_files():
    # File A (0, 2, 4) and File B (1, 3, 5) interleaved in physical offsets
    # Physical order: A0 (0), B0 (1), A1 (2), B1 (3), A2 (4), B2 (5)
    frags = [
        _make_fragment("A0", 0, b"\xff\xd8\xff\xe0" + b"A" * 508),
        _make_fragment("B0", 512, b"\x89PNG\r\n\x1a\n" + b"B" * 504),
        _make_fragment("A1", 1024, b"\xaa" * 512),
        _make_fragment("B1", 1536, b"\xbb" * 512),
        _make_fragment("A2", 2048, b"A" * 500 + b"\xff\xd9\x00\x00\x00\x00\x00\x00\x00\x00"),
        _make_fragment("B2", 2560, b"B" * 500 + b"IEND\xaeB`\x82"),
    ]

    edges = [
        (0, 2), (2, 4),  # Correct A transitions
        (1, 3), (3, 5),  # Correct B transitions
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # False physical proximity edges
    ]
    p_next = {
        (0, 2): 0.95, (2, 4): 0.92,
        (1, 3): 0.94, (3, 5): 0.91,
        (0, 1): 0.10, (1, 2): 0.10, (2, 3): 0.10, (3, 4): 0.10, (4, 5): 0.10,
    }
    p_same = {
        (0, 2): 0.98, (2, 4): 0.98,
        (1, 3): 0.97, (3, 5): 0.97,
        (0, 1): 0.05, (1, 2): 0.05, (2, 3): 0.05, (3, 4): 0.05, (4, 5): 0.05,
    }

    optimizer = GlobalConstrainedReconstructionOptimizer(OptimizerConfig(file_penalty=1.0))
    output = optimizer.reconstruct(frags, edges, p_next, p_same)

    assert output.solver_result.status == "OPTIMAL"
    assert len(output.reconstructed_files) == 2
    seqs = [rf.fragment_sequence for rf in output.reconstructed_files]
    assert ["A0", "A1", "A2"] in seqs
    assert ["B0", "B1", "B2"] in seqs


# -------------------------------------------------------------------------
# Test 4: Fragmented non-contiguous files
# -------------------------------------------------------------------------
def test_04_fragmented_non_contiguous_files():
    # Out of order: Frag 2, Frag 0, Frag 3, Frag 1
    frags = [
        _make_fragment("F2", 0, b"DATA_PART_2_" + b"\x22" * 500),
        _make_fragment("F0", 512, b"\xff\xd8\xff\xe0" + b"DATA_PART_0_" + b"\x00" * 496),
        _make_fragment("F3", 1024, b"DATA_PART_3_" + b"\x33" * 488 + b"\xff\xd9\x00\x00\x00\x00\x00\x00\x00\x00"),
        _make_fragment("F1", 1536, b"DATA_PART_1_" + b"\x11" * 500),
    ]
    # Logical order: F0 (idx 1) -> F1 (idx 3) -> F2 (idx 0) -> F3 (idx 2)
    edges = [
        (1, 3), (3, 0), (0, 2),
        (0, 1), (2, 3),  # Wrong physical edges
    ]
    p_next = {
        (1, 3): 0.96, (3, 0): 0.93, (0, 2): 0.95,
        (0, 1): 0.10, (2, 3): 0.08,
    }
    p_same = {e: 0.99 for e in edges}

    optimizer = GlobalConstrainedReconstructionOptimizer(OptimizerConfig(file_penalty=0.8))
    output = optimizer.reconstruct(frags, edges, p_next, p_same)

    assert output.solver_result.status == "OPTIMAL"
    assert len(output.reconstructed_files) == 1
    assert output.reconstructed_files[0].fragment_sequence == ["F0", "F1", "F2", "F3"]


# -------------------------------------------------------------------------
# Test 5: Missing intermediate fragment (gap variable activation)
# -------------------------------------------------------------------------
def test_05_missing_intermediate_fragment():
    # F0 (header) -> [F1 is lost] -> F2 (footer)
    frags = [
        _make_fragment("F0", 0, b"\xff\xd8\xff\xe0" + b"\x00" * 508),
        _make_fragment("F2", 8192, b"\x00" * 500 + b"\xff\xd9\x00\x00\x00\x00\x00\x00\x00\x00"),
    ]
    # Direct transition is weak due to discontinuity, but same-file is high
    edges = [(0, 1)]
    p_next = {(0, 1): 0.05}
    p_same = {(0, 1): 0.95}

    optimizer = GlobalConstrainedReconstructionOptimizer(
        OptimizerConfig(enable_gaps=True, base_gap_penalty=1.0, file_penalty=0.5)
    )
    output = optimizer.reconstruct(frags, edges, p_next, p_same)

    assert output.solver_result.status == "OPTIMAL"
    assert len(output.reconstructed_files) == 1
    seq = output.reconstructed_files[0].fragment_sequence
    # Expect gap marker between F0 and F2
    assert "F0" in seq
    assert "F2" in seq
    assert len(seq) == 3
    assert "[GAP_F0_F2]" == seq[1]


# -------------------------------------------------------------------------
# Test 6: Corrupted fragment handling
# -------------------------------------------------------------------------
def test_06_corrupted_fragment():
    # Sequence with a corrupted fragment with low affinity
    frags = [
        _make_fragment("F0", 0, b"\x89PNG\r\n\x1a\n" + b"\x00" * 504),
        _make_fragment("FCORRUPT", 512, b"\xde\xad\xbe\xef" * 128),
        _make_fragment("F2", 1024, b"\x00" * 500 + b"IEND\xaeB`\x82"),
    ]
    edges = [(0, 1), (1, 2), (0, 2)]
    # F0->FCORRUPT and FCORRUPT->F2 have poor predictions
    p_next = {(0, 1): 0.05, (1, 2): 0.05, (0, 2): 0.85}
    p_same = {(0, 1): 0.10, (1, 2): 0.10, (0, 2): 0.90}

    optimizer = GlobalConstrainedReconstructionOptimizer(OptimizerConfig(file_penalty=1.0))
    output = optimizer.reconstruct(frags, edges, p_next, p_same)

    assert output.solver_result.status == "OPTIMAL"
    # Solver should route around corrupted fragment or exclude it
    seqs = [rf.fragment_sequence for rf in output.reconstructed_files]
    assert any("F0" in s and "F2" in s for s in seqs)
    # FCORRUPT should not be inside the main reconstructed chain
    main_chain = next(s for s in seqs if "F0" in s)
    assert "FCORRUPT" not in main_chain


# -------------------------------------------------------------------------
# Test 7: Contradictory high-scoring edges (forking resolution)
# -------------------------------------------------------------------------
def test_07_contradictory_high_scoring_edges():
    # Frag 0 has high next probability to both Frag 1 and Frag 2
    frags = [
        _make_fragment("F0", 0, b"\xaa" * 512),
        _make_fragment("F1", 512, b"\xbb" * 512),
        _make_fragment("F2", 1024, b"\xcc" * 512),
    ]
    edges = [(0, 1), (0, 2)]
    p_next = {(0, 1): 0.95, (0, 2): 0.91}
    p_same = {(0, 1): 0.98, (0, 2): 0.95}

    optimizer = GlobalConstrainedReconstructionOptimizer(OptimizerConfig(file_penalty=0.5))
    output = optimizer.reconstruct(frags, edges, p_next, p_same)

    assert output.solver_result.status == "OPTIMAL"
    # Out-degree <= 1 must hold strictly! Exactly one successor selected for F0.
    selected = output.solver_result.selected_edges
    out_edges_0 = [e for e in selected if e[0] == 0]
    assert len(out_edges_0) == 1
    assert out_edges_0[0] == (0, 1)  # Higher scoring edge won


# -------------------------------------------------------------------------
# Test 8: Directed cycle elimination (MTZ subtour constraints)
# -------------------------------------------------------------------------
def test_08_directed_cycle_elimination():
    # Complete directed cycle: 0 -> 1 -> 2 -> 0
    frags = [
        _make_fragment("C0", 0, b"CYCLE_0_" + b"\x00" * 504),
        _make_fragment("C1", 512, b"CYCLE_1_" + b"\x00" * 504),
        _make_fragment("C2", 1024, b"CYCLE_2_" + b"\x00" * 504),
    ]
    edges = [(0, 1), (1, 2), (2, 0)]
    # All edges have very high individual scores
    p_next = {(0, 1): 0.95, (1, 2): 0.95, (2, 0): 0.95}
    p_same = {e: 0.99 for e in edges}

    # With MTZ ordering enabled
    optimizer = GlobalConstrainedReconstructionOptimizer(
        OptimizerConfig(enable_mtz_ordering=True, file_penalty=0.5)
    )
    output = optimizer.reconstruct(frags, edges, p_next, p_same)

    assert output.solver_result.status == "OPTIMAL"
    selected = output.solver_result.selected_edges
    # A cycle of length 3 would require all 3 edges; MTZ forbids this, so at most 2 edges can be selected
    assert len(selected) <= 2
    # Check graph acyclicity
    assert not ( (0, 1) in selected and (1, 2) in selected and (2, 0) in selected )


# -------------------------------------------------------------------------
# Test 9: Duplicate fragment usage prevention (in-degree <= 1)
# -------------------------------------------------------------------------
def test_09_duplicate_fragment_usage_prevention():
    # Two candidates both competing to be followed by fragment 2
    frags = [
        _make_fragment("F0", 0, b"\xaa" * 512),
        _make_fragment("F1", 512, b"\xbb" * 512),
        _make_fragment("F2", 1024, b"\xcc" * 512),
    ]
    edges = [(0, 2), (1, 2)]
    p_next = {(0, 2): 0.88, (1, 2): 0.94}
    p_same = {e: 0.95 for e in edges}

    optimizer = GlobalConstrainedReconstructionOptimizer(OptimizerConfig(file_penalty=0.5))
    output = optimizer.reconstruct(frags, edges, p_next, p_same)

    assert output.solver_result.status == "OPTIMAL"
    selected = output.solver_result.selected_edges
    in_edges_2 = [e for e in selected if e[1] == 2]
    assert len(in_edges_2) == 1
    assert in_edges_2[0] == (1, 2)  # Higher scoring predecessor won


# -------------------------------------------------------------------------
# Test 10: Infeasible constraint handling
# -------------------------------------------------------------------------
def test_10_infeasible_constraint_handling():
    # Construct an inherently infeasible system:
    # x_01 forced to be both >= 1.0 and <= 0.0
    vmap = OptimizationVariableBuilder.build_variable_map(2, [(0, 1)])
    c_builder = ConstraintMatrixBuilder(vmap)
    c_builder.add_constraint({0: 1.0}, lb=1.0, ub=1.0)
    c_builder.add_constraint({0: 1.0}, lb=0.0, ub=0.0)

    A, lhs, rhs = c_builder.get_scipy_sparse_constraints()
    obj_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][:vmap.total_variables])

    solver = MILPReconstructionSolver(time_limit_seconds=5.0)
    res = solver.solve(vmap, obj_weights, A, lhs, rhs)

    # Must report INFEASIBLE gracefully without crashing or throwing unhandled exception
    assert res.status == "INFEASIBLE"
    assert res.success is False
