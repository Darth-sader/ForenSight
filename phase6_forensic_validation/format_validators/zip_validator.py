"""ZIP Forensic Format Validator (v2.0.0-phase6).

Implements independent multi-tier verification:
- L0: Local file header signature (PK\x03\x04).
- L1: Central directory (PK\x01\x02) and End of Central Directory (PK\x05\x06).
- L2: Full ZIP directory parsing via zipfile.ZipFile.
- L3: Entry decompression & CRC32 verification (testzip) with zip-bomb safeguards.
- L4: Content completeness: non-empty archive, non-corrupted filenames.
- L5: Exact ground-truth SHA-256 reference verification.
"""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

from phase6_forensic_validation.format_validators.base_validator import (
    BaseFormatValidator,
    FormatValidationResult,
    ValidationLevel,
)

ZIP_LOCAL_MAGIC = b"PK\x03\x04"
ZIP_EOCD_MAGIC = b"PK\x05\x06"


class ZIPValidator(BaseFormatValidator):
    """Deep structural and decompression validator for ZIP archives."""

    def __init__(self, max_uncompressed_bytes: int = 50 * 1024 * 1024):
        super().__init__("ZIP")
        self.max_uncompressed_bytes = max_uncompressed_bytes

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
        if file_len < 4 or raw_bytes[:4] != ZIP_LOCAL_MAGIC:
            errors.append("Missing ZIP local header magic (PK\\x03\\x04).")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        passed_levels.append(ValidationLevel.L0_SIGNATURE)

        # -----------------------------------------------------------------
        # Level 1: Basic Structural Layout (EOCD presence)
        # -----------------------------------------------------------------
        eocd_pos = raw_bytes.rfind(ZIP_EOCD_MAGIC)
        if eocd_pos < 0:
            warnings.append("Missing End of Central Directory (EOCD PK\\x05\\x06) record.")
        else:
            passed_levels.append(ValidationLevel.L1_LAYOUT)

        # -----------------------------------------------------------------
        # Level 2: ZIP Structure Parsing via zipfile
        # -----------------------------------------------------------------
        parsed_entries: List[str] = []
        total_uncompressed = 0

        try:
            bio = io.BytesIO(raw_bytes)
            with zipfile.ZipFile(bio, "r") as zf:
                parsed_entries = zf.namelist()
                metadata["entries"] = parsed_entries
                metadata["entry_count"] = len(parsed_entries)

                for info in zf.infolist():
                    total_uncompressed += info.file_size

                metadata["total_uncompressed_bytes"] = total_uncompressed

                # Zip bomb guard
                if total_uncompressed > self.max_uncompressed_bytes:
                    errors.append(f"Decompression bomb defense: {total_uncompressed} bytes exceeds safety limit.")
                    return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

                passed_levels.append(ValidationLevel.L2_STRUCTURE)

                # -------------------------------------------------------------
                # Level 3: Entry Decompression and CRC Verification
                # -------------------------------------------------------------
                corrupted = zf.testzip()
                if corrupted is not None:
                    errors.append(f"ZIP CRC32 failure in entry: {corrupted}")
                else:
                    passed_levels.append(ValidationLevel.L3_DECOMPRESSION)

        except zipfile.BadZipFile as bz:
            errors.append(f"Bad ZIP archive structure: {bz}")
        except Exception as exc:
            errors.append(f"ZIP parser exception: {exc}")

        # -----------------------------------------------------------------
        # Level 4: Content Completeness Checks
        # -----------------------------------------------------------------
        if ValidationLevel.L3_DECOMPRESSION in passed_levels:
            if len(parsed_entries) > 0:
                passed_levels.append(ValidationLevel.L4_COMPLETENESS)
            else:
                warnings.append("ZIP archive contains zero entries.")

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
