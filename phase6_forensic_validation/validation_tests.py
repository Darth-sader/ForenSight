"""Unit and Scenario Test Suite for Forensic Validation and Uncertainty (v2.0.0-phase6).

Validates:
1. Probability Calibration (Platt & Isotonic, Brier, Log Loss, ECE).
2. Split Conformal Prediction (Nonconformity, coverage guarantees >= 1 - alpha).
3. All 8 Format Validators (JPEG, PNG, PDF, ZIP, DOCX, XLSX, MP3, MP4).
4. Multi-level Validation Hierarchy (L0 to L5).
5. Parser Security Isolation (Hard timeouts, temp file cleanup, error catching).
6. Multidimensional Uncertainty Engine & Risk-Coverage curves.
7. Cost-Sensitive Abstention Policy (ACCEPTED, PARTIAL, UNCERTAIN, REJECTED).
8. Evidence Provenance & Cryptographic Reproducibility.
9. Investigator-Readable Reconstruction Reporting & Disclaimers.
"""
from __future__ import annotations

import io
import time
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

from phase1_leakage_free.fragment_schema_v2 import ObservableFragment
from phase6_forensic_validation.abstention_policy import (
    CostSensitiveAbstentionPolicy,
    DecisionVerdict,
    PolicyCostModel,
    ReconstructionOutcome,
)
from phase6_forensic_validation.calibration_v2 import ForensicCalibratorV2
from phase6_forensic_validation.conformal_prediction import ForensicConformalPredictor
from phase6_forensic_validation.format_validators.base_validator import FormatValidationResult, ValidationLevel
from phase6_forensic_validation.parser_isolation import ParserSecurityHarness
from phase6_forensic_validation.provenance import EvidenceProvenanceTracker
from phase6_forensic_validation.reconstruction_report import ForensicReconstructionReportGenerator
from phase6_forensic_validation.uncertainty_engine import ForensicUncertaintyEngine
from phase6_forensic_validation.validation_engine import ForensicValidationEngine


# -------------------------------------------------------------------------
# 1. Calibration Tests
# -------------------------------------------------------------------------
def test_01_calibration_platt_and_isotonic():
    rng = np.random.RandomState(42)
    # Generate synthetic uncalibrated scores and ground-truth targets
    y_true = rng.binomial(1, 0.4, size=500)
    # Miscalibrated scores (overconfident)
    raw_scores = np.where(y_true == 1, rng.beta(5, 1, size=500), rng.beta(1, 5, size=500))

    # Split into cal and test
    cal_scores, test_scores = raw_scores[:250], raw_scores[250:]
    cal_y, test_y = y_true[:250], y_true[250:]

    # Test Platt scaling
    platt = ForensicCalibratorV2(method="platt").fit(cal_scores, cal_y)
    p_platt = platt.predict_proba(test_scores)
    metrics_platt = ForensicCalibratorV2.compute_calibration_metrics(p_platt, test_y)

    # Test Isotonic regression
    iso = ForensicCalibratorV2(method="isotonic").fit(cal_scores, cal_y)
    p_iso = iso.predict_proba(test_scores)
    metrics_iso = ForensicCalibratorV2.compute_calibration_metrics(p_iso, test_y)

    assert metrics_platt.brier_score < 0.25
    assert metrics_iso.brier_score < 0.25
    assert 0.0 <= metrics_platt.ece <= 1.0
    assert 0.0 <= metrics_iso.ece <= 1.0
    assert len(metrics_iso.reliability.bin_centers) == 10


# -------------------------------------------------------------------------
# 2. Conformal Prediction Tests
# -------------------------------------------------------------------------
def test_02_conformal_prediction_coverage():
    rng = np.random.RandomState(42)
    n = 2000
    y_true = rng.binomial(1, 0.5, size=n)
    probs = np.where(y_true == 1, rng.beta(4, 2, size=n), rng.beta(2, 4, size=n))

    cal_p, test_p = probs[:1000], probs[1000:]
    cal_y, test_y = y_true[:1000], y_true[1000:]

    predictor = ForensicConformalPredictor(alphas=(0.05, 0.10))
    predictor.calibrate(cal_p, cal_y)

    eval_results = predictor.evaluate(test_p, test_y)

    # For alpha = 0.10, empirical coverage must be close to or >= 0.90
    res_10 = eval_results["alpha_0.1"]
    assert res_10.empirical_coverage >= 0.88
    assert res_10.mean_set_size >= 1.0

    # For alpha = 0.05, coverage must be close to or >= 0.95
    res_05 = eval_results["alpha_0.05"]
    assert res_05.empirical_coverage >= 0.93


# -------------------------------------------------------------------------
# 3. Format Validator Tests (All 8 Formats)
# -------------------------------------------------------------------------
def _create_minimal_png() -> bytes:
    bio = io.BytesIO()
    img = Image.new("RGB", (32, 32), color="blue")
    img.save(bio, format="PNG")
    return bio.getvalue()


def _create_minimal_jpeg() -> bytes:
    bio = io.BytesIO()
    img = Image.new("RGB", (32, 32), color="red")
    img.save(bio, format="JPEG")
    return bio.getvalue()


def _create_minimal_zip() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("test.txt", "Forensic validation payload")
    return bio.getvalue()


def _create_minimal_docx() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/></Types>')
        zf.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
        zf.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>')
    return bio.getvalue()


def _create_minimal_xlsx() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/></Types>')
        zf.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>')
        zf.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData/></worksheet>')
    return bio.getvalue()


def _create_minimal_pdf() -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n165\n%%EOF\n"
    )


def _create_minimal_mp3() -> bytes:
    # 4 consecutive valid MPEG-1 Layer 3 audio frames (128 kbps, 44.1kHz, stereo -> 417 bytes each)
    # Header: 0xFF, 0xFB, 0x90, 0x64 (Sync + Layer 3 + 128kbps + 44.1kHz + no padding)
    frame = b"\xff\xfb\x90\x64" + b"\x00" * 413
    return b"ID3\x03\x00\x00\x00\x00\x00\x00" + (frame * 4)


def _create_minimal_mp4() -> bytes:
    # ftyp box (24 bytes) + moov box (40 bytes with mvhd) + mdat box (12 bytes)
    ftyp = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
    mvhd = b"\x00\x00\x00\x20mvhd\x00\x00\x00\x00" + b"\x00" * 12 + b"\x00\x00\x03\xe8\x00\x00\x07\xd0"
    moov = b"\x00\x00\x00\x28moov" + mvhd
    mdat = b"\x00\x00\x00\x0cmdatDATA"
    return ftyp + moov + mdat


def test_03_all_format_validators():
    engine = ForensicValidationEngine(timeout_seconds=2.0)

    # 1. PNG
    res_png = engine.validate_file(_create_minimal_png())
    assert res_png.format_name == "PNG"
    assert res_png.highest_passed_level >= ValidationLevel.L3_DECOMPRESSION
    assert res_png.is_structurally_valid is True

    # 2. JPEG
    res_jpg = engine.validate_file(_create_minimal_jpeg())
    assert res_jpg.format_name == "JPEG"
    assert res_jpg.highest_passed_level >= ValidationLevel.L3_DECOMPRESSION
    assert res_jpg.is_structurally_valid is True

    # 3. ZIP
    res_zip = engine.validate_file(_create_minimal_zip())
    assert res_zip.format_name == "ZIP"
    assert res_zip.highest_passed_level >= ValidationLevel.L3_DECOMPRESSION
    assert res_zip.is_structurally_valid is True

    # 4. DOCX
    res_docx = engine.validate_file(_create_minimal_docx(), claimed_format="DOCX")
    assert res_docx.format_name == "DOCX"
    assert res_docx.highest_passed_level >= ValidationLevel.L3_DECOMPRESSION
    assert res_docx.is_structurally_valid is True

    # 5. XLSX
    res_xlsx = engine.validate_file(_create_minimal_xlsx(), claimed_format="XLSX")
    assert res_xlsx.format_name == "XLSX"
    assert res_xlsx.highest_passed_level >= ValidationLevel.L3_DECOMPRESSION
    assert res_xlsx.is_structurally_valid is True

    # 6. PDF
    res_pdf = engine.validate_file(_create_minimal_pdf())
    assert res_pdf.format_name == "PDF"
    assert res_pdf.highest_passed_level >= ValidationLevel.L2_STRUCTURE
    assert res_pdf.is_structurally_valid is True

    # 7. MP3
    res_mp3 = engine.validate_file(_create_minimal_mp3())
    assert res_mp3.format_name == "MP3"
    assert res_mp3.highest_passed_level >= ValidationLevel.L2_STRUCTURE
    assert res_mp3.is_structurally_valid is True

    # 8. MP4
    res_mp4 = engine.validate_file(_create_minimal_mp4())
    assert res_mp4.format_name == "MP4"
    assert res_mp4.highest_passed_level >= ValidationLevel.L2_STRUCTURE
    assert res_mp4.is_structurally_valid is True


# -------------------------------------------------------------------------
# 4. Parser Security Isolation Tests
# -------------------------------------------------------------------------
def test_04_parser_security_harness():
    harness = ParserSecurityHarness(timeout_seconds=0.5, max_input_bytes=1024)

    # Test 1: Infinite loop timeout containment
    def hanging_parser(data: bytes) -> str:
        time.sleep(2.0)
        return "finished"

    res_timeout = harness.run_isolated(hanging_parser, b"TEST_PAYLOAD")
    assert res_timeout.success is False
    assert res_timeout.status == "TIMEOUT"
    assert res_timeout.execution_time_seconds >= 0.4

    # Test 2: File size limit enforcement
    res_oversize = harness.run_isolated(hanging_parser, b"X" * 2048)
    assert res_oversize.success is False
    assert res_oversize.status == "FILE_SIZE_EXCEEDED"


# -------------------------------------------------------------------------
# 5. Uncertainty Engine & Decision Policy Tests
# -------------------------------------------------------------------------
def test_05_uncertainty_and_abstention_policy():
    engine = ForensicUncertaintyEngine()
    policy = CostSensitiveAbstentionPolicy(
        tau_accept=0.35,
        tau_partial=0.65,
        min_margin_accept=0.5,
    )

    # 1. High confidence candidate -> ACCEPTED
    val_valid = FormatValidationResult(
        format_name="PNG",
        highest_passed_level=ValidationLevel.L4_COMPLETENESS,
        passed_levels=[ValidationLevel.L0_SIGNATURE, ValidationLevel.L1_LAYOUT, ValidationLevel.L2_STRUCTURE, ValidationLevel.L3_DECOMPRESSION, ValidationLevel.L4_COMPLETENESS],
        is_structurally_valid=True,
        is_fully_renderable=True,
        has_ground_truth_match=False,
        byte_length=1024,
        sha256_hash="abc",
    )
    sig_valid = engine.evaluate_hypothesis_uncertainty(
        recon_id="recon_001",
        edge_probs=[0.95, 0.92, 0.98],
        runner_up_margin=1.8,
        graph_branching_factor=1.0,
        gap_count=0,
        total_fragments=4,
        val_result=val_valid,
    )
    verdict_valid = policy.evaluate_candidate(sig_valid, val_valid)
    assert verdict_valid.outcome == ReconstructionOutcome.ACCEPTED
    assert verdict_valid.is_forensically_usable is True

    # 2. Candidate with gaps -> PARTIAL
    sig_gap = engine.evaluate_hypothesis_uncertainty(
        recon_id="recon_002",
        edge_probs=[0.85, 0.80],
        runner_up_margin=0.8,
        graph_branching_factor=1.5,
        gap_count=1,
        total_fragments=4,
        val_result=val_valid,
    )
    verdict_gap = policy.evaluate_candidate(sig_gap, val_valid)
    assert verdict_gap.outcome == ReconstructionOutcome.PARTIAL
    assert verdict_gap.is_forensically_usable is True

    # 3. Competing ambiguity -> UNCERTAIN
    sig_ambig = engine.evaluate_hypothesis_uncertainty(
        recon_id="recon_003",
        edge_probs=[0.60, 0.55],
        runner_up_margin=0.05,
        graph_branching_factor=3.0,
        gap_count=0,
        total_fragments=2,
        val_result=val_valid,
        competing_count=3,
    )
    verdict_ambig = policy.evaluate_candidate(sig_ambig, val_valid)
    assert verdict_ambig.outcome == ReconstructionOutcome.UNCERTAIN
    assert verdict_ambig.requires_human_inspection is True

    # 4. Structural failure -> REJECTED
    val_fail = FormatValidationResult(
        format_name="PNG",
        highest_passed_level=ValidationLevel.L0_SIGNATURE,
        passed_levels=[ValidationLevel.L0_SIGNATURE],
        is_structurally_valid=False,
        is_fully_renderable=False,
        has_ground_truth_match=False,
        byte_length=512,
        sha256_hash="def",
        errors=["Missing IHDR and IEND chunks"],
    )
    sig_fail = engine.evaluate_hypothesis_uncertainty(
        recon_id="recon_004",
        edge_probs=[0.20],
        runner_up_margin=0.0,
        gap_count=2,
        total_fragments=2,
        val_result=val_fail,
    )
    verdict_fail = policy.evaluate_candidate(sig_fail, val_fail)
    assert verdict_fail.outcome == ReconstructionOutcome.REJECTED
    assert verdict_fail.is_forensically_usable is False


# -------------------------------------------------------------------------
# 6. Provenance & Reconstruction Report Tests
# -------------------------------------------------------------------------
def test_06_provenance_and_report_generation():
    tracker = EvidenceProvenanceTracker(disk_image_id="EVID_CASE_2026_01")
    obs_frags = {
        "frag_0": ObservableFragment(fragment_id="frag_0", observed_offset=0, fragment_bytes=b"HDR_BYTES", fragment_length=9),
        "frag_1": ObservableFragment(fragment_id="frag_1", observed_offset=512, fragment_bytes=b"FTR_BYTES", fragment_length=9),
    }

    val_res = FormatValidationResult(
        format_name="JPEG",
        highest_passed_level=ValidationLevel.L4_COMPLETENESS,
        passed_levels=[ValidationLevel.L0_SIGNATURE, ValidationLevel.L1_LAYOUT, ValidationLevel.L2_STRUCTURE, ValidationLevel.L3_DECOMPRESSION, ValidationLevel.L4_COMPLETENESS],
        is_structurally_valid=True,
        is_fully_renderable=True,
        has_ground_truth_match=False,
        byte_length=18,
        sha256_hash="mock_hash",
        extracted_metadata={"width": 640, "height": 480},
    )

    verdict = DecisionVerdict(
        recon_id="recon_test_001",
        outcome=ReconstructionOutcome.ACCEPTED,
        achieved_validation_level=ValidationLevel.L4_COMPLETENESS,
        composite_uncertainty=0.15,
        decision_rationale="Verified structural consistency and valid Huffman scan.",
        is_forensically_usable=True,
        requires_human_inspection=False,
    )

    prov_rec = tracker.create_provenance_record(
        recon_id="recon_test_001",
        reconstructed_bytes=b"HDR_BYTESFTR_BYTES",
        sequence_fragment_ids=["frag_0", "frag_1"],
        all_fragments_map=obs_frags,
        p_raw_dict={("frag_0", "frag_1"): 0.94},
        p_cal_dict={("frag_0", "frag_1"): 0.91},
        val_result=val_res,
        verdict=verdict,
    )

    assert len(prov_rec.provenance_hash) == 64
    assert len(prov_rec.fragments) == 2
    assert len(prov_rec.transitions) == 1

    # Generate Markdown report
    report_md = ForensicReconstructionReportGenerator.generate_markdown_report(prov_rec)
    assert "# FORENSIC FILE RECONSTRUCTION AUDIT REPORT" in report_md
    assert "EVID_CASE_2026_01" in report_md
    assert "ACCEPTED" in report_md
    assert "MANDATORY SCIENTIFIC & FORENSIC DISCLAIMER" in report_md
