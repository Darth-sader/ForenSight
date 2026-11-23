"""JPEG Forensic Format Validator (v2.0.0-phase6).

Implements independent multi-tier verification:
- L0: SOI marker (0xFFD8) presence.
- L1: SOI + EOI (0xFFD9) boundary check and minimum header layout.
- L2: Structural segment parser: DQT, SOF0/SOF2, DHT, SOS length integrity.
- L3: Image decompression test via PIL (Image.open / verify / load).
- L4: Image completeness: dimensions > 0, valid color components (1, 3, 4).
- L5: Exact ground-truth SHA-256 reference verification.
"""
from __future__ import annotations

import hashlib
import io
import struct
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from phase6_forensic_validation.format_validators.base_validator import (
    BaseFormatValidator,
    FormatValidationResult,
    ValidationLevel,
)


class JPEGValidator(BaseFormatValidator):
    """Deep structural and decoder validator for JPEG image streams."""

    def __init__(self):
        super().__init__("JPEG")

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
        # Level 0: Signature Recognition (SOI)
        # -----------------------------------------------------------------
        if file_len < 4 or raw_bytes[:2] != b"\xff\xd8":
            errors.append("Missing JPEG Start of Image (SOI 0xFFD8) marker at byte 0.")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        passed_levels.append(ValidationLevel.L0_SIGNATURE)

        # -----------------------------------------------------------------
        # Level 1: Basic Structural Layout (SOI + EOI within file)
        # -----------------------------------------------------------------
        eoi_pos = raw_bytes.rfind(b"\xff\xd9")
        if eoi_pos < 0:
            warnings.append("Missing JPEG End of Image (EOI 0xFFD9) marker in payload.")
        elif eoi_pos < file_len - 16:
            warnings.append(f"EOI marker found at offset {eoi_pos}, but file continues to {file_len} (slack/trailing data).")
            passed_levels.append(ValidationLevel.L1_LAYOUT)
        else:
            passed_levels.append(ValidationLevel.L1_LAYOUT)

        # -----------------------------------------------------------------
        # Level 2: Structural Marker & Segment Parsing
        # -----------------------------------------------------------------
        idx = 2
        dqt_count = 0
        dht_count = 0
        sof_found = False
        sos_found = False
        width, height, components = 0, 0, 0

        try:
            while idx < file_len - 1:
                # Seek next marker
                if raw_bytes[idx] != 0xFF:
                    idx += 1
                    continue

                marker = raw_bytes[idx + 1]
                idx += 2

                # Ignore padding bytes
                if marker in (0x00, 0xFF):
                    continue

                # Standalone markers without length
                if marker in (0xD8, 0xD9, 0x01) or (0xD0 <= marker <= 0xD7):
                    if marker == 0xD9:  # EOI
                        break
                    continue

                if idx + 2 > file_len:
                    errors.append(f"Truncated segment length for marker 0xFF{marker:02X} at offset {idx}")
                    break

                seg_len = struct.unpack(">H", raw_bytes[idx:idx + 2])[0]
                if seg_len < 2 or idx + seg_len > file_len:
                    errors.append(f"Invalid segment length {seg_len} at offset {idx}")
                    break

                seg_payload = raw_bytes[idx + 2:idx + seg_len]

                if marker == 0xDB:  # DQT (Quantization Table)
                    dqt_count += 1
                elif marker == 0xC4:  # DHT (Huffman Table)
                    dht_count += 1
                elif marker in (0xC0, 0xC1, 0xC2):  # SOF (Start of Frame)
                    sof_found = True
                    if len(seg_payload) >= 6:
                        precision, h, w, comp = struct.unpack(">BHHB", seg_payload[:6])
                        height, width, components = h, w, comp
                elif marker == 0xDA:  # SOS (Start of Scan)
                    sos_found = True
                    # SOS is followed immediately by entropy-coded image data
                    break

                idx += seg_len

            metadata["dqt_tables"] = dqt_count
            metadata["dht_tables"] = dht_count
            metadata["sof_found"] = sof_found
            metadata["sos_found"] = sos_found
            metadata["width"] = width
            metadata["height"] = height
            metadata["components"] = components

            if sof_found and (dqt_count > 0 or sos_found):
                passed_levels.append(ValidationLevel.L2_STRUCTURE)
            else:
                errors.append("Structural parsing incomplete: missing essential SOF or DQT tables.")

        except Exception as exc:
            errors.append(f"Segment parser exception: {exc}")

        # -----------------------------------------------------------------
        # Level 3: Decoder Acceptance (PIL Load Test)
        # -----------------------------------------------------------------
        if ValidationLevel.L2_STRUCTURE in passed_levels:
            try:
                bio = io.BytesIO(raw_bytes)
                with Image.open(bio) as img:
                    img.verify()

                # Full decode test
                bio.seek(0)
                with Image.open(bio) as img:
                    img.load()
                    pil_w, pil_h = img.size
                    metadata["pil_size"] = [pil_w, pil_h]
                    metadata["pil_mode"] = img.mode

                passed_levels.append(ValidationLevel.L3_DECOMPRESSION)
            except Exception as exc:
                errors.append(f"PIL JPEG decompression failed: {exc}")

        # -----------------------------------------------------------------
        # Level 4: Content Completeness Checks
        # -----------------------------------------------------------------
        if ValidationLevel.L3_DECOMPRESSION in passed_levels:
            if width > 0 and height > 0 and components in (1, 3, 4):
                passed_levels.append(ValidationLevel.L4_COMPLETENESS)
            else:
                warnings.append("Dimensions or component counts non-standard.")

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
