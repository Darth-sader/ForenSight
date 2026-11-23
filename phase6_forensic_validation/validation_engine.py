"""Multi-Format Validation Engine with Process Isolation (v2.0.0-phase6).

Coordinates deep structural validation across all 8 forensic formats:
- JPEG, PNG, PDF, ZIP, DOCX, XLSX, MP3, MP4.
- Automatically selects appropriate validator based on signature or claimed type.
- Executes parsing inside ParserSecurityHarness with hard timeouts and decompression limits.
- Evaluates multi-tier validation levels (L0 to L5).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from phase6_forensic_validation.format_validators.base_validator import (
    BaseFormatValidator,
    FormatValidationResult,
    ValidationLevel,
)
from phase6_forensic_validation.format_validators.docx_validator import DOCXValidator
from phase6_forensic_validation.format_validators.jpeg_validator import JPEGValidator
from phase6_forensic_validation.format_validators.mp3_validator import MP3Validator
from phase6_forensic_validation.format_validators.mp4_validator import MP4Validator
from phase6_forensic_validation.format_validators.pdf_validator import PDFValidator
from phase6_forensic_validation.format_validators.png_validator import PNGValidator
from phase6_forensic_validation.format_validators.xlsx_validator import XLSXValidator
from phase6_forensic_validation.format_validators.zip_validator import ZIPValidator
from phase6_forensic_validation.parser_isolation import ParserExecutionResult, ParserSecurityHarness


class ForensicValidationEngine:
    """Master forensic validation coordinator with security isolation."""

    def __init__(self, timeout_seconds: float = 3.0):
        self.security_harness = ParserSecurityHarness(timeout_seconds=timeout_seconds)
        self.validators: Dict[str, BaseFormatValidator] = {
            "JPEG": JPEGValidator(),
            "PNG": PNGValidator(),
            "PDF": PDFValidator(),
            "ZIP": ZIPValidator(),
            "DOCX": DOCXValidator(),
            "XLSX": XLSXValidator(),
            "MP3": MP3Validator(),
            "MP4": MP4Validator(),
        }

    def detect_format(self, raw_bytes: bytes) -> Optional[str]:
        """Detect file format based on initial magic signatures."""
        if len(raw_bytes) < 4:
            return None

        if raw_bytes[:2] == b"\xff\xd8":
            return "JPEG"
        elif len(raw_bytes) >= 8 and raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "PNG"
        elif b"%PDF-" in raw_bytes[:1024]:
            return "PDF"
        elif raw_bytes[:4] == b"PK\x03\x04":
            # Disambiguate ZIP vs DOCX vs XLSX
            if b"word/document.xml" in raw_bytes or b"word/" in raw_bytes:
                return "DOCX"
            elif b"xl/workbook.xml" in raw_bytes or b"xl/" in raw_bytes:
                return "XLSX"
            return "ZIP"
        elif raw_bytes[:3] == b"ID3" or (raw_bytes[0] == 0xFF and (raw_bytes[1] & 0xE0) == 0xE0):
            return "MP3"
        elif b"ftyp" in raw_bytes[:64]:
            return "MP4"

        return None

    def validate_file(
        self,
        raw_bytes: bytes,
        claimed_format: Optional[str] = None,
        reference_hash: Optional[str] = None,
    ) -> FormatValidationResult:
        """Validate a reconstructed byte payload under security isolation."""
        target_fmt = claimed_format.upper() if claimed_format else self.detect_format(raw_bytes)
        if not target_fmt or target_fmt not in self.validators:
            # Fallback signature detection
            target_fmt = self.detect_format(raw_bytes)

        if not target_fmt or target_fmt not in self.validators:
            # Unknown / Unsupported format
            return FormatValidationResult(
                format_name=claimed_format or "UNKNOWN",
                highest_passed_level=ValidationLevel.L0_SIGNATURE,
                passed_levels=[],
                is_structurally_valid=False,
                is_fully_renderable=False,
                has_ground_truth_match=False,
                byte_length=len(raw_bytes),
                sha256_hash="",
                errors=["Unable to detect supported file format or signature."],
                warnings=[],
                extracted_metadata={},
            )

        validator = self.validators[target_fmt]

        # Execute parser within isolated security harness
        exec_res: ParserExecutionResult = self.security_harness.run_isolated(
            validator.validate,
            raw_bytes,
            reference_hash=reference_hash,
        )

        if exec_res.success and isinstance(exec_res.output_data, FormatValidationResult):
            return exec_res.output_data
        else:
            # Handle security/isolation aborts (timeout, bomb, error)
            return FormatValidationResult(
                format_name=target_fmt,
                highest_passed_level=ValidationLevel.L0_SIGNATURE,
                passed_levels=[],
                is_structurally_valid=False,
                is_fully_renderable=False,
                has_ground_truth_match=False,
                byte_length=len(raw_bytes),
                sha256_hash=exec_res.input_hash,
                errors=[f"Validation aborted by security harness: {exec_res.status} ({exec_res.error_message})"],
                warnings=[],
                extracted_metadata={"security_status": exec_res.status},
                execution_time_seconds=exec_res.execution_time_seconds,
            )
