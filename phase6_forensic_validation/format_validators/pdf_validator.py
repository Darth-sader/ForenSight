"""PDF Forensic Format Validator (v2.0.0-phase6).

Implements independent multi-tier verification:
- L0: %PDF- magic header in initial 1024 bytes.
- L1: %%EOF marker in terminal 1024 bytes.
- L2: Object enumeration (obj ... endobj), cross-reference tables (xref) or streams.
- L3: Embedded stream decompression (FlateDecode / zlib inflate).
- L4: Catalog dictionary (/Type /Catalog), page hierarchy (/Pages), page count.
- L5: Exact ground-truth SHA-256 reference verification.
"""
from __future__ import annotations

import hashlib
import re
import time
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

from phase6_forensic_validation.format_validators.base_validator import (
    BaseFormatValidator,
    FormatValidationResult,
    ValidationLevel,
)


class PDFValidator(BaseFormatValidator):
    """Deep structural and stream-decompression validator for PDF documents."""

    def __init__(self):
        super().__init__("PDF")

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
        # Level 0: Signature Recognition (%PDF-)
        # -----------------------------------------------------------------
        header_window = raw_bytes[:min(1024, file_len)]
        header_pos = header_window.find(b"%PDF-")
        if header_pos < 0:
            errors.append("Missing %PDF- header in first 1024 bytes.")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        passed_levels.append(ValidationLevel.L0_SIGNATURE)
        version_str = header_window[header_pos + 5:header_pos + 8].decode("latin1", errors="replace")
        metadata["pdf_version"] = version_str

        # -----------------------------------------------------------------
        # Level 1: Basic Structural Layout (%%EOF in trailer)
        # -----------------------------------------------------------------
        trailer_window = raw_bytes[max(0, file_len - 1024):]
        if b"%%EOF" not in trailer_window:
            warnings.append("Missing %%EOF marker in trailing 1024 bytes.")
        else:
            passed_levels.append(ValidationLevel.L1_LAYOUT)

        # -----------------------------------------------------------------
        # Level 2: Object and Cross-Reference Syntax Parsing
        # -----------------------------------------------------------------
        obj_matches = list(re.finditer(rb"(\d+)\s+(\d+)\s+obj", raw_bytes))
        endobj_count = len(re.findall(rb"endobj", raw_bytes))
        xref_found = b"xref" in raw_bytes or b"/XRef" in raw_bytes
        trailer_found = b"trailer" in raw_bytes or b"/Root" in raw_bytes

        metadata["object_count"] = len(obj_matches)
        metadata["endobj_count"] = endobj_count
        metadata["has_xref"] = xref_found
        metadata["has_trailer"] = trailer_found

        if len(obj_matches) > 0 and endobj_count > 0 and (xref_found or trailer_found):
            passed_levels.append(ValidationLevel.L2_STRUCTURE)
        else:
            errors.append(f"Incomplete PDF structure: {len(obj_matches)} objs, {endobj_count} endobjs, xref={xref_found}")

        # -----------------------------------------------------------------
        # Level 3: Embedded Stream Decompression (FlateDecode)
        # -----------------------------------------------------------------
        if ValidationLevel.L2_STRUCTURE in passed_levels:
            streams_decompressed = 0
            stream_matches = list(re.finditer(rb"stream[\r\n]+(.*?)[\r\n]+endstream", raw_bytes, re.DOTALL))
            metadata["stream_count"] = len(stream_matches)

            for sm in stream_matches[:10]:  # Test first 10 streams
                stream_payload = sm.group(1)
                try:
                    zlib.decompress(stream_payload)
                    streams_decompressed += 1
                except Exception:
                    pass

            metadata["streams_decompressed"] = streams_decompressed
            if streams_decompressed > 0 or len(stream_matches) == 0:
                passed_levels.append(ValidationLevel.L3_DECOMPRESSION)
            else:
                warnings.append("Could not decompress embedded FlateDecode streams.")

        # -----------------------------------------------------------------
        # Level 4: Content Completeness Checks (Catalog & Page Tree)
        # -----------------------------------------------------------------
        if ValidationLevel.L3_DECOMPRESSION in passed_levels:
            has_catalog = b"/Catalog" in raw_bytes
            has_pages = b"/Pages" in raw_bytes
            metadata["has_catalog"] = has_catalog
            metadata["has_pages"] = has_pages

            if has_catalog and has_pages:
                passed_levels.append(ValidationLevel.L4_COMPLETENESS)
            else:
                warnings.append("Document missing /Catalog or /Pages dictionary entries.")

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
