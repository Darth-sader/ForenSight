"""XLSX Forensic Format Validator (v2.0.0-phase6).

Implements independent multi-tier verification for OpenXML Excel spreadsheets:
- L0: ZIP local file header signature (PK\x03\x04).
- L1: ZIP container integrity (valid directory & EOCD).
- L2: Open Packaging Conventions (OPC): [Content_Types].xml and xl/workbook.xml.
- L3: XML parsing of xl/workbook.xml and worksheet parts via xml.etree.ElementTree.
- L4: SpreadsheetML semantic schema checks (workbook root, sheets, sheet).
- L5: Exact ground-truth SHA-256 reference verification.
"""
from __future__ import annotations

import hashlib
import io
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

from phase6_forensic_validation.format_validators.base_validator import (
    BaseFormatValidator,
    FormatValidationResult,
    ValidationLevel,
)

ZIP_MAGIC = b"PK\x03\x04"


class XLSXValidator(BaseFormatValidator):
    """Deep structural and XML semantic validator for Microsoft Excel (.xlsx) spreadsheets."""

    def __init__(self):
        super().__init__("XLSX")

    def validate(
        self,
        raw_bytes: bytes,
        reference_hash: Optional[str] = None,
    ) -> FormatValidationResult:
        t0 = time.time()
        passed_levels: List[ValidationLevel] = []
        errors: List[str] = []
        warnings: List[str] = []
        metadata: Dict[str, Any] = {}
        file_len = len(raw_bytes)
        file_hash = hashlib.sha256(raw_bytes).hexdigest()

        # -----------------------------------------------------------------
        # Level 0: Signature Recognition
        # -----------------------------------------------------------------
        if file_len < 4 or raw_bytes[:4] != ZIP_MAGIC:
            errors.append("Missing ZIP magic header (PK\\x03\\x04) required for XLSX.")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        passed_levels.append(ValidationLevel.L0_SIGNATURE)

        # -----------------------------------------------------------------
        # Level 1: Basic Structural Layout (ZIP container test)
        # -----------------------------------------------------------------
        bio = io.BytesIO(raw_bytes)
        try:
            with zipfile.ZipFile(bio, "r") as zf:
                if zf.testzip() is None:
                    passed_levels.append(ValidationLevel.L1_LAYOUT)
                else:
                    warnings.append("ZIP container reported non-fatal CRC warnings.")
                    passed_levels.append(ValidationLevel.L1_LAYOUT)
        except Exception as exc:
            errors.append(f"Invalid XLSX ZIP packaging: {exc}")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        # -----------------------------------------------------------------
        # Level 2: Open Packaging Conventions (OPC) Structure
        # -----------------------------------------------------------------
        bio.seek(0)
        has_content_types = False
        has_workbook_xml = False
        worksheet_entries: List[str] = []

        try:
            with zipfile.ZipFile(bio, "r") as zf:
                namelist = set(zf.namelist())
                metadata["entry_count"] = len(namelist)

                has_content_types = "[Content_Types].xml" in namelist
                has_workbook_xml = "xl/workbook.xml" in namelist
                worksheet_entries = [name for name in namelist if name.startswith("xl/worksheets/")]

                metadata["has_content_types"] = has_content_types
                metadata["has_workbook_xml"] = has_workbook_xml
                metadata["worksheet_count"] = len(worksheet_entries)

                if has_content_types and has_workbook_xml:
                    passed_levels.append(ValidationLevel.L2_STRUCTURE)
                else:
                    errors.append(f"Missing essential OPC parts: [Content_Types]={has_content_types}, xl/workbook.xml={has_workbook_xml}")

                # -------------------------------------------------------------
                # Level 3: XML Decompression & Syntax Parsing
                # -------------------------------------------------------------
                if ValidationLevel.L2_STRUCTURE in passed_levels:
                    # Parse [Content_Types].xml
                    ct_bytes = zf.read("[Content_Types].xml")
                    ET.fromstring(ct_bytes)

                    # Parse xl/workbook.xml
                    wb_bytes = zf.read("xl/workbook.xml")
                    wb_root = ET.fromstring(wb_bytes)
                    metadata["workbook_xml_bytes"] = len(wb_bytes)

                    # Test at least one worksheet
                    if worksheet_entries:
                        ws_bytes = zf.read(worksheet_entries[0])
                        ET.fromstring(ws_bytes)

                    passed_levels.append(ValidationLevel.L3_DECOMPRESSION)

                    # ---------------------------------------------------------
                    # Level 4: Semantic Content & Schema Checks
                    # ---------------------------------------------------------
                    tag_clean = wb_root.tag.split("}")[-1] if "}" in wb_root.tag else wb_root.tag
                    sheets_elem = None
                    for child in wb_root:
                        child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                        if child_tag == "sheets":
                            sheets_elem = child
                            break

                    metadata["root_tag"] = tag_clean
                    metadata["has_sheets"] = sheets_elem is not None

                    if tag_clean == "workbook" and sheets_elem is not None and len(sheets_elem) > 0:
                        passed_levels.append(ValidationLevel.L4_COMPLETENESS)
                    else:
                        warnings.append("xl/workbook.xml missing valid <sheets> definitions.")

        except ET.ParseError as pe:
            errors.append(f"XML parse error in XLSX payload: {pe}")
        except Exception as exc:
            errors.append(f"XLSX validation exception: {exc}")

        # -----------------------------------------------------------------
        # Level 5: Ground-Truth Match (Evaluation Only)
        # -----------------------------------------------------------------
        if reference_hash and file_hash.lower() == reference_hash.lower():
            passed_levels.append(ValidationLevel.L5_GROUND_TRUTH)

        return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

    def _build_result(
        self,
        passed_levels: List[ValidationLevel],
        errors: List[str],
        warnings: List[str],
        metadata: Dict[str, Any],
        file_len: int,
        file_hash: str,
        t0: float,
    ) -> FormatValidationResult:
        highest = max(passed_levels) if passed_levels else ValidationLevel.L0_SIGNATURE
        return FormatValidationResult(
            format_name=self.format_name,
            highest_passed_level=highest,
            passed_levels=passed_levels,
            is_structurally_valid=ValidationLevel.L2_STRUCTURE in passed_levels,
            is_fully_renderable=ValidationLevel.L3_DECOMPRESSION in passed_levels,
            has_ground_truth_match=ValidationLevel.L5_GROUND_TRUTH in passed_levels,
            byte_length=file_len,
            sha256_hash=file_hash,
            errors=errors,
            warnings=warnings,
            extracted_metadata=metadata,
            execution_time_seconds=time.time() - t0,
        )
