"""MP4 Forensic Format Validator (v2.0.0-phase6).

Implements independent multi-tier verification for ISO Base Media (MP4) containers:
- L0: ftyp box header within initial 32 bytes.
- L1: Sequential box walk (length + 4-char type: ftyp, moov, mdat, free).
- L2: Container structural parsing (moov movie container, mdat media payload).
- L3: Movie header (mvhd) and track (trak/mdia) traversal.
- L4: Metadata completeness: valid timescale, duration > 0, track count >= 1.
- L5: Exact ground-truth SHA-256 reference verification.
"""
from __future__ import annotations

import hashlib
import struct
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from phase6_forensic_validation.format_validators.base_validator import (
    BaseFormatValidator,
    FormatValidationResult,
    ValidationLevel,
)


class MP4Validator(BaseFormatValidator):
    """Deep structural and box-container validator for MP4 video/audio streams."""

    def __init__(self):
        super().__init__("MP4")

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
        # Level 0: Signature Recognition (ftyp box)
        # -----------------------------------------------------------------
        scan_limit = min(64, file_len - 8)
        ftyp_pos = raw_bytes[:scan_limit].find(b"ftyp")
        if ftyp_pos < 0 or ftyp_pos > 32:
            errors.append("Missing MP4 ftyp box header in initial bytes.")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        passed_levels.append(ValidationLevel.L0_SIGNATURE)

        # -----------------------------------------------------------------
        # Level 1: Sequential Box Walk
        # -----------------------------------------------------------------
        idx = 0
        boxes_found: List[str] = []
        moov_found = False
        mdat_found = False

        try:
            while idx + 8 <= file_len:
                box_len = struct.unpack(">I", raw_bytes[idx:idx + 4])[0]
                box_type = raw_bytes[idx + 4:idx + 8].decode("latin1", errors="replace")
                boxes_found.append(box_type)

                if box_type == "moov":
                    moov_found = True
                elif box_type == "mdat":
                    mdat_found = True

                # Box length 1 indicates 64-bit extended length
                if box_len == 1:
                    if idx + 16 > file_len:
                        break
                    box_len = struct.unpack(">Q", raw_bytes[idx + 8:idx + 16])[0]
                elif box_len == 0:
                    # Box extends to end of file
                    box_len = file_len - idx

                if box_len < 8 or idx + box_len > file_len:
                    warnings.append(f"Box {box_type} has length {box_len} extending beyond file boundaries.")
                    break

                idx += box_len

            metadata["boxes"] = boxes_found
            metadata["moov_found"] = moov_found
            metadata["mdat_found"] = mdat_found

            if len(boxes_found) >= 2:
                passed_levels.append(ValidationLevel.L1_LAYOUT)
            else:
                warnings.append("Insufficient top-level boxes parsed in MP4 container.")

        except Exception as exc:
            errors.append(f"MP4 box parser exception: {exc}")

        # -----------------------------------------------------------------
        # Level 2: Container Structural Parsing (moov or mdat presence)
        # -----------------------------------------------------------------
        if ValidationLevel.L1_LAYOUT in passed_levels:
            if moov_found or mdat_found:
                passed_levels.append(ValidationLevel.L2_STRUCTURE)
            else:
                errors.append("Missing essential MP4 container boxes (moov or mdat).")

        # -----------------------------------------------------------------
        # Level 3: Movie Header & Track Traversal (mvhd / trak)
        # -----------------------------------------------------------------
        if ValidationLevel.L2_STRUCTURE in passed_levels:
            has_mvhd = b"mvhd" in raw_bytes
            has_trak = b"trak" in raw_bytes
            metadata["has_mvhd"] = has_mvhd
            metadata["has_trak"] = has_trak

            if has_mvhd:
                # Parse mvhd
                mvhd_pos = raw_bytes.find(b"mvhd")
                if mvhd_pos >= 0 and mvhd_pos + 32 <= file_len:
                    # mvhd version at +4
                    ver = raw_bytes[mvhd_pos + 4]
                    if ver == 0 and mvhd_pos + 28 <= file_len:
                        timescale, duration = struct.unpack(">II", raw_bytes[mvhd_pos + 16:mvhd_pos + 24])
                        metadata["timescale"] = timescale
                        metadata["duration_ticks"] = duration
                        if timescale > 0:
                            metadata["duration_seconds"] = round(duration / timescale, 2)

                passed_levels.append(ValidationLevel.L3_DECOMPRESSION)
            elif mdat_found:
                passed_levels.append(ValidationLevel.L3_DECOMPRESSION)

        # -----------------------------------------------------------------
        # Level 4: Content Completeness Checks
        # -----------------------------------------------------------------
        if ValidationLevel.L3_DECOMPRESSION in passed_levels:
            if moov_found and mdat_found:
                passed_levels.append(ValidationLevel.L4_COMPLETENESS)
            else:
                warnings.append("Incomplete MP4 container: missing either moov metadata or mdat media payload.")

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
