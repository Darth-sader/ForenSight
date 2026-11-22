"""Unit Tests for Dual-Relation Learning and Reconstruction (v2.0.0-phase3).

Verifies:
1. Same-file symmetry: P_same(i, j) == P_same(j, i)
2. Directed adjacency asymmetry: P_next(i, j) != P_next(j, i)
3. Non-identity assertion: P_same != P_next (rejects copy/reuse vulnerability)
4. Reverse adjacency property: P_prev(i, j) == P_next(j, i)
5. Graph edge attribute completeness: All 9 mandated attributes present on edges
6. Multi-task shared encoder gradient backpropagation
"""
from __future__ import annotations

import sys
from pathlib import Path

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import networkx as nx
import numpy as np
import pytest
import torch

from phase3_dual_relation.dual_relation_graph import (
    DualRelationReconstructionEngine,
    build_dual_relation_graph,
)
from phase3_dual_relation.dual_relation_inference import (
    DualRelationInferenceEngine,
    ProbabilityReuseError,
)
from phase3_dual_relation.multitask_model import MultiTaskDualModel, MultiTaskNet
from phase3_dual_relation.next_fragment_model_v2 import NextFragmentModelV2
from phase3_dual_relation.same_file_model_v2 import SameFileModelV2


@pytest.fixture
def synthetic_pair_data():
    """Generates synthetic forward and reverse feature pairs for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 362

    # Forward features
    X_fwd = np.random.randn(n_samples, n_features).astype(np.float32)

    # Reverse features (simulate asymmetric directional features)
    X_rev = X_fwd.copy()
    # Boundary features (asymmetric) differ in reverse
    X_rev[:, :20] = np.random.randn(n_samples, 20).astype(np.float32)

    # Synthetic labels
    y_same = np.random.randint(0, 2, size=n_samples)
    y_next = np.random.randint(0, 2, size=n_samples)

    return X_fwd, X_rev, y_same, y_next


def test_same_file_symmetry(synthetic_pair_data):
    """Test 1: P_same(i, j) == P_same(j, i) must hold to machine precision."""
    X_fwd, X_rev, y_same, _ = synthetic_pair_data

    model = SameFileModelV2(model_type="random_forest", n_estimators=10, max_depth=4, seed=42)
    model.fit(X_fwd, y_same)

    # Compute symmetric probabilities for forward pair (i, j) and reverse pair (j, i)
    p_cal_ij, p_raw_ij = model.predict_proba_symmetric(X_fwd, X_rev)
    p_cal_ji, p_raw_ji = model.predict_proba_symmetric(X_rev, X_fwd)

    # Must be identical regardless of pair order
    np.testing.assert_allclose(p_cal_ij, p_cal_ji, atol=1e-6, err_msg="P_same violated symmetry!")
    np.testing.assert_allclose(p_raw_ij, p_raw_ji, atol=1e-6, err_msg="Raw P_same violated symmetry!")


def test_next_fragment_asymmetry(synthetic_pair_data):
    """Test 2: P_next(i, j) is directional and asymmetric in general."""
    X_fwd, X_rev, _, y_next = synthetic_pair_data

    model = NextFragmentModelV2(model_type="random_forest", n_estimators=15, max_depth=4, seed=42)
    model.fit(X_fwd, y_next)

    p_next_cal, p_prev_cal, p_next_raw, p_prev_raw = model.predict_proba_directed(X_fwd, X_rev)

    # Forward and reverse transitions must diverge on asymmetric features
    divergence = np.mean(np.abs(p_next_cal - p_prev_cal))
    assert divergence > 1e-4, f"P_next failed to exhibit directional asymmetry! Divergence: {divergence}"


def test_reverse_adjacency_property(synthetic_pair_data):
    """Test 3: P_prev(i, j) == P_next(j, i)."""
    X_fwd, X_rev, _, y_next = synthetic_pair_data

    model = NextFragmentModelV2(model_type="random_forest", n_estimators=15, max_depth=4, seed=42)
    model.fit(X_fwd, y_next)

    # P_prev(i, j) computed from forward call
    _, p_prev_ij, _, _ = model.predict_proba_directed(X_fwd, X_rev)

    # P_next(j, i) computed by evaluating reverse features as forward
    p_next_ji = model.predict_proba(X_rev)

    np.testing.assert_allclose(p_prev_ij, p_next_ji, atol=1e-6,
                              err_msg="P_prev(i, j) != P_next(j, i) consistency failed!")


def test_non_identity_check():
    """Test 4: DualRelationInferenceEngine detects and rejects probability reuse."""
    class LeakingModel:
        def predict_proba_symmetric(self, fwd, rev):
            # Returns fixed mock probabilities
            p = np.full(len(fwd), 0.75)
            return p, p

    class CopyingNextModel:
        def predict_proba_directed(self, fwd, rev):
            # Returns exact same probability as same-file model
            p = np.full(len(fwd), 0.75)
            return p, p, p, p

    engine = DualRelationInferenceEngine(
        same_model=LeakingModel(),
        next_model=CopyingNextModel(),
    )

    X_mock = np.ones((10, 362), dtype=np.float32)

    with pytest.raises(ProbabilityReuseError, match="Detected illegal probability reuse / copying"):
        engine.compute_pair_scores(X_mock, X_mock, validate_non_identity=True)


def test_graph_edge_attributes():
    """Test 5: Every directed candidate edge must retain all 9 mandated forensic attributes."""
    mandated_attrs = {
        "source",
        "target",
        "p_same",
        "p_next",
        "p_prev",
        "embedding_sim",
        "boundary_compat",
        "provenance",
        "model_version",
    }

    fragment_ids = [f"F{i:03d}" for i in range(5)]
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 2]])
    p_same = np.array([0.9, 0.85, 0.88, 0.92, 0.4])
    p_next = np.array([0.95, 0.80, 0.75, 0.70, 0.1])
    p_prev = np.array([0.05, 0.08, 0.02, 0.01, 0.0])
    emb_sim = np.array([0.91, 0.87, 0.83, 0.89, 0.3])
    b_compat = np.array([0.8, 0.7, 0.6, 0.75, 0.2])
    provenance = [{"origin": "test", "pair_id": i} for i in range(5)]

    graph = build_dual_relation_graph(
        fragment_ids=fragment_ids,
        successor_pairs=pairs,
        p_same=p_same,
        p_next=p_next,
        p_prev=p_prev,
        embedding_sim=emb_sim,
        boundary_compat=b_compat,
        provenance=provenance,
        model_version="2.0.0-phase3-test",
    )

    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 5

    for u, v, data in graph.edges(data=True):
        for attr in mandated_attrs:
            assert attr in data, f"Mandated attribute '{attr}' missing on edge ({u} -> {v})!"


def test_multitask_shared_encoder_gradient_flow():
    """Test 6: Gradients from both task loss heads backpropagate into shared encoder trunk."""
    net = MultiTaskNet(input_dim=64, shared_dim=32, head_hidden=16)

    x_sym = torch.randn(8, 64, requires_grad=True)
    x_dir = torch.randn(8, 64, requires_grad=True)
    y_same = torch.randint(0, 2, (8,)).float()
    y_next = torch.randint(0, 2, (8,)).float()

    logits_same, logits_next = net(x_sym, x_dir)

    criterion = torch.nn.BCEWithLogitsLoss()
    loss_same = criterion(logits_same, y_same)
    loss_next = criterion(logits_next, y_next)
    joint_loss = 1.0 * loss_same + 1.0 * loss_next

    joint_loss.backward()

    # Check gradients in shared trunk
    trunk_has_grads = False
    for param in net.shared_trunk.parameters():
        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            trunk_has_grads = True
            break

    assert trunk_has_grads, "Shared encoder trunk received zero gradients from joint loss!"


def test_reconstruction_engine_dual_stages():
    """Test 7: DualRelationReconstructionEngine clusters by P_same and orders by P_next."""
    fragment_ids = ["F0", "F1", "F2", "F3"]
    # F0, F1 from File 1 (high p_same); F2, F3 from File 2 (high p_same); cross-file low p_same
    pairs = np.array([
        [0, 1],  # F0 -> F1 (same file 1, next)
        [1, 0],
        [2, 3],  # F2 -> F3 (same file 2, next)
        [3, 2],
        [0, 2],  # cross file (low p_same)
        [1, 3],
    ])
    p_same = np.array([0.95, 0.95, 0.92, 0.92, 0.05, 0.08])
    p_next = np.array([0.98, 0.01, 0.94, 0.02, 0.05, 0.04])
    p_prev = np.array([0.01, 0.98, 0.02, 0.94, 0.01, 0.02])

    graph = build_dual_relation_graph(
        fragment_ids=fragment_ids,
        successor_pairs=pairs,
        p_same=p_same,
        p_next=p_next,
        p_prev=p_prev,
    )

    frag_pool = {
        "F0": b"AAAA" * 128,
        "F1": b"BBBB" * 128,
        "F2": b"CCCC" * 128,
        "F3": b"DDDD" * 128,
    }

    engine = DualRelationReconstructionEngine(tau_same=0.5, tau_next=0.3)
    recons = engine.reconstruct(graph, frag_pool, mode="variant_c_dual_independent")

    # Should produce 2 distinct reconstructed files (F0 -> F1, and F2 -> F3)
    assert len(recons) == 2, f"Expected 2 partitioned files, got {len(recons)}"
    seqs = [r.fragment_sequence for r in recons]
    assert ["F0", "F1"] in seqs
    assert ["F2", "F3"] in seqs
