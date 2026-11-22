"""Unit and Known-Answer Tests for Forensic Evaluation Engine (v2.0.0-phase2).

Validates the mathematical correctness and edge-case handling of:
- Exact byte-perfect reconstruction rate
- Partial reconstruction rate (LCS >= 0.5)
- Byte recovery ratio
- False file merge and false file split rates
- Successor precision, recall, and F1
- Fragment placement accuracy
- Duplicate assignment rate
- Unused fragment rate
"""
import pytest
import hashlib

from phase2_dataset_evaluation.evaluation_engine import (
    EvaluationEngine,
    ReconstructedFile,
    sha256_bytes,
)


@pytest.fixture
def synthetic_gt_manifest():
    """Build a known-answer ground-truth manifest with 2 files, 4 fragments each."""
    f1_bytes = b"AAAA" * 128 + b"BBBB" * 128 + b"CCCC" * 128 + b"DDDD" * 128  # 2048 bytes (4 frags)
    f2_bytes = b"EEEE" * 128 + b"FFFF" * 128 + b"GGGG" * 128 + b"HHHH" * 128  # 2048 bytes (4 frags)

    manifest = {
        "files": {
            "doc1.pdf": {
                "source_file_id": "doc1.pdf",
                "original_file_hash": sha256_bytes(f1_bytes),
                "file_size_bytes": len(f1_bytes),
                "total_fragments": 4,
                "file_type": "PDF",
                "logical_byte_offsets": [0, 512, 1024, 1536],
                "logical_fragment_sequence": [0, 1, 2, 3],
                "fragment_hashes": [
                    sha256_bytes(f1_bytes[i:i+512]) for i in range(0, len(f1_bytes), 512)
                ],
                "observed_physical_locations": [512, 1024, 1536, 2048],
            },
            "img2.jpg": {
                "source_file_id": "img2.jpg",
                "original_file_hash": sha256_bytes(f2_bytes),
                "file_size_bytes": len(f2_bytes),
                "total_fragments": 4,
                "file_type": "JPEG",
                "logical_byte_offsets": [0, 512, 1024, 1536],
                "logical_fragment_sequence": [0, 1, 2, 3],
                "fragment_hashes": [
                    sha256_bytes(f2_bytes[i:i+512]) for i in range(0, len(f2_bytes), 512)
                ],
                "observed_physical_locations": [2560, 3072, 3584, 4096],
            },
        }
    }
    return manifest, f1_bytes, f2_bytes


def test_evaluator_perfect_reconstruction(synthetic_gt_manifest):
    """Test Case 1: 100% exact byte-perfect recovery."""
    manifest, f1_bytes, f2_bytes = synthetic_gt_manifest
    engine = EvaluationEngine()

    recon_files = [
        ReconstructedFile(
            recon_id="rf1",
            fragment_sequence=["doc1.pdf_frag_0000", "doc1.pdf_frag_0001", "doc1.pdf_frag_0002", "doc1.pdf_frag_0003"],
            reconstructed_bytes=f1_bytes,
        ),
        ReconstructedFile(
            recon_id="rf2",
            fragment_sequence=["img2.jpg_frag_0000", "img2.jpg_frag_0001", "img2.jpg_frag_0002", "img2.jpg_frag_0003"],
            reconstructed_bytes=f2_bytes,
        ),
    ]

    all_frags = {
        "doc1.pdf_frag_0000", "doc1.pdf_frag_0001", "doc1.pdf_frag_0002", "doc1.pdf_frag_0003",
        "img2.jpg_frag_0000", "img2.jpg_frag_0001", "img2.jpg_frag_0002", "img2.jpg_frag_0003",
    }

    results = engine.evaluate_reconstruction(manifest, recon_files, all_pool_fragment_ids=all_frags)

    assert results["file_level"]["exact_reconstruction_rate"] == 1.0
    assert results["file_level"]["partial_reconstruction_rate"] == 0.0  # Exact overrides partial
    assert results["file_level"]["byte_recovery_ratio"] == 1.0
    assert results["file_level"]["false_file_merge_rate"] == 0.0
    assert results["file_level"]["false_file_split_rate"] == 0.0
    assert results["fragment_level"]["fragment_placement_accuracy"] == 1.0
    assert results["fragment_level"]["successor_precision"] == 1.0
    assert results["fragment_level"]["successor_recall"] == 1.0
    assert results["fragment_level"]["duplicate_assignment_rate"] == 0.0
    assert results["fragment_level"]["unused_fragment_rate"] == 0.0


def test_evaluator_missing_fragments_partial_recovery(synthetic_gt_manifest):
    """Test Case 2: Missing fragments resulting in partial recovery without exact match."""
    manifest, f1_bytes, _ = synthetic_gt_manifest
    engine = EvaluationEngine()

    # Reconstruct doc1.pdf with fragment 2 missing (has frag 0, 1, 3: 3 out of 4 = 75% LCS)
    f1_partial_bytes = f1_bytes[:1024] + f1_bytes[1536:]
    recon_files = [
        ReconstructedFile(
            recon_id="rf1",
            fragment_sequence=["doc1.pdf_frag_0000", "doc1.pdf_frag_0001", "doc1.pdf_frag_0003"],
            reconstructed_bytes=f1_partial_bytes,
        )
    ]

    results = engine.evaluate_reconstruction(manifest, recon_files)

    # 0% exact recovery because hash does not match
    assert results["file_level"]["exact_reconstruction_rate"] == 0.0
    # 50% partial recovery (1 of 2 GT files recovered partially)
    assert results["file_level"]["partial_reconstruction_rate"] == 0.5
    # Byte recovery ratio: 1536 / 4096 = 0.375
    assert results["file_level"]["byte_recovery_ratio"] == 0.375
    # Successor precision: 1 transition (0->1) is valid, 1 (1->3) is invalid -> 1/2 = 0.5
    assert results["fragment_level"]["successor_precision"] == 0.5


def test_evaluator_false_file_merge(synthetic_gt_manifest):
    """Test Case 3: Chimeric file merge combining fragments from two distinct original files."""
    manifest, f1_bytes, f2_bytes = synthetic_gt_manifest
    engine = EvaluationEngine()

    chimeric_bytes = f1_bytes[:1024] + f2_bytes[:1024]
    recon_files = [
        ReconstructedFile(
            recon_id="rf_chimeric",
            fragment_sequence=["doc1.pdf_frag_0000", "doc1.pdf_frag_0001", "img2.jpg_frag_0000", "img2.jpg_frag_0001"],
            reconstructed_bytes=chimeric_bytes,
        )
    ]

    results = engine.evaluate_reconstruction(manifest, recon_files)

    assert results["file_level"]["false_file_merge_rate"] == 1.0  # 1 of 1 file is merged
    assert results["file_level"]["exact_reconstruction_rate"] == 0.0


def test_evaluator_duplicate_fragment_assignment(synthetic_gt_manifest):
    """Test Case 4: Duplicate fragment assigned to two distinct files."""
    manifest, f1_bytes, f2_bytes = synthetic_gt_manifest
    engine = EvaluationEngine()

    recon_files = [
        ReconstructedFile(
            recon_id="rf1",
            fragment_sequence=["doc1.pdf_frag_0000", "doc1.pdf_frag_0001"],
            reconstructed_bytes=f1_bytes[:1024],
        ),
        ReconstructedFile(
            recon_id="rf2",
            fragment_sequence=["doc1.pdf_frag_0000", "img2.jpg_frag_0000"],
            reconstructed_bytes=b"XXXX",
        ),
    ]

    results = engine.evaluate_reconstruction(manifest, recon_files)

    # 4 assignments, 3 unique fragments -> 1 duplicate -> 1/4 = 0.25
    assert results["fragment_level"]["duplicate_assignment_rate"] == 0.25


def test_evaluator_incorrect_fragment_ordering(synthetic_gt_manifest):
    """Test Case 5: Reverse fragment ordering (0% exact match, 0 valid successors)."""
    manifest, f1_bytes, _ = synthetic_gt_manifest
    engine = EvaluationEngine()

    rev_bytes = f1_bytes[1536:] + f1_bytes[1024:1536] + f1_bytes[512:1024] + f1_bytes[:512]
    recon_files = [
        ReconstructedFile(
            recon_id="rf_rev",
            fragment_sequence=["doc1.pdf_frag_0003", "doc1.pdf_frag_0002", "doc1.pdf_frag_0001", "doc1.pdf_frag_0000"],
            reconstructed_bytes=rev_bytes,
        )
    ]

    results = engine.evaluate_reconstruction(manifest, recon_files)

    assert results["file_level"]["exact_reconstruction_rate"] == 0.0
    assert results["fragment_level"]["successor_precision"] == 0.0
    assert results["fragment_level"]["successor_recall"] == 0.0
    # Placements: only index matching position gets counted (none match true positions 3->0, 2->1, 1->2, 0->3)
    assert results["fragment_level"]["fragment_placement_accuracy"] == 0.0
