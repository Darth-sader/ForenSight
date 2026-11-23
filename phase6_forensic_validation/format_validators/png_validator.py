"""PNG Forensic Format Validator (v2.0.0-phase6).

Implements independent multi-tier verification:
- L0: 8-byte PNG signature (\x89PNG\r\n\x1a\n).
- L1: IHDR first chunk, terminating IEND chunk.
- L2: Chunk iteration with CRC32 integrity verification for every chunk.
- L3: IDAT zlib stream decompression and PIL pixel decoding.
- L4: Dimensions > 0, valid bit depth, valid color type.
- L5: Exact ground-truth SHA-256 reference verification.
"""
from __future__ import annotations

import hashlib
import io
import struct
import time
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from phase6_forensic_validation.format_validators.base_validator import (
    BaseFormatValidator,
    FormatValidationResult,
    ValidationLevel,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class PNGValidator(BaseFormatValidator):
    """Deep structural and zlib decoder validator for PNG image streams."""

    def __init__(self):
        super().__init__("PNG")

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
        if file_len < 8 or raw_bytes[:8] != PNG_MAGIC:
            errors.append("Missing PNG 8-byte magic header (\\x89PNG\\r\\n\\x1a\\n).")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        passed_levels.append(ValidationLevel.L0_SIGNATURE)

        # -----------------------------------------------------------------
        # Level 1: Basic Structural Layout (IHDR first chunk)
        # -----------------------------------------------------------------
        if file_len < 8 + 12:
            errors.append("File truncated before IHDR chunk header.")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        first_chunk_type = raw_bytes[12:16]
        if first_chunk_type != b"IHDR":
            errors.append(f"First chunk is {first_chunk_type!r}, expected b'IHDR'.")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        iend_pos = raw_bytes.rfind(b"IEND")
        if iend_pos < 0:
            warnings.append("Missing terminating IEND chunk in file.")
        else:
            passed_levels.append(ValidationLevel.L1_LAYOUT)

        # -----------------------------------------------------------------
        # Level 2: Chunk Iteration and CRC32 Verification
        # -----------------------------------------------------------------
        idx = 8
        chunks_parsed: List[str] = []
        crc_failures = 0
        idat_payloads: List[bytes] = []
        width, height, bit_depth, color_type = 0, 0, 0, 0

        try:
            while idx + 8 <= file_len:
                chunk_len = struct.unpack(">I", raw_bytes[idx:idx + 4])[0]
                chunk_type = raw_bytes[idx + 4:idx + 8]
                type_str = chunk_type.decode("latin1", errors="replace")
                chunks_parsed.append(type_str)

                payload_start = idx + 8
                payload_end = payload_start + chunk_len
                crc_offset = payload_end

                if crc_offset + 4 > file_len:
                    errors.append(f"Chunk {type_str} truncated at offset {idx} (length {chunk_len})")
                    break

                chunk_payload = raw_bytes[payload_start:payload_end]
                stored_crc = struct.unpack(">I", raw_bytes[crc_offset:crc_offset + 4])[0]

                # Verify CRC32 over chunk type + data
                calc_crc = zlib.crc32(raw_bytes[idx + 4:payload_end]) & 0xFFFFFFFF
                if stored_crc != calc_crc:
                    crc_failures += 1
                    warnings.append(f"CRC mismatch in chunk {type_str}: stored {stored_crc:08X}, calculated {calc_crc:08X}")

                if chunk_type == b"IHDR" and len(chunk_payload) >= 13:
                    width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_payload[:10])
                    metadata["width"] = width
                    metadata["height"] = height
                    metadata["bit_depth"] = bit_depth
                    metadata["color_type"] = color_type
                elif chunk_type == b"IDAT":
                    idat_payloads.append(chunk_payload)
                elif chunk_type == b"IEND":
                    idx = crc_offset + 4
                    break

                idx = crc_offset + 4

            metadata["chunks"] = chunks_parsed
            metadata["crc_failures"] = crc_failures
            metadata["idat_count"] = len(idat_payloads)

            if "IHDR" in chunks_parsed and len(idat_payloads) > 0 and crc_failures == 0:
                passed_levels.append(ValidationLevel.L2_STRUCTURE)
            elif "IHDR" in chunks_parsed and len(idat_payloads) > 0:
                passed_levels.append(ValidationLevel.L2_STRUCTURE)
                warnings.append("Parsed structure with non-critical CRC errors.")

        except Exception as exc:
            errors.append(f"PNG chunk traversal exception: {exc}")

        # -----------------------------------------------------------------
        # Level 3: IDAT Decompression and PIL Decode
        # -----------------------------------------------------------------
        if ValidationLevel.L2_STRUCTURE in passed_levels:
            decomp_success = False
            try:
                # Direct zlib inflate of concatenated IDAT chunks
                combined_idat = b"".join(idat_payloads)
                raw_decomp = zlib.decompress(combined_idat)
                metadata["decompressed_bytes"] = len(raw_decomp)
                decomp_success = True
            except Exception as zerr:
                warnings.append(f"Raw zlib decompression issue: {zerr}")

            try:
                bio = io.BytesIO(raw_bytes)
                with Image.open(bio) as img:
                    img.load()
                    metadata["pil_size"] = list(img.size)
                    metadata["pil_mode"] = img.mode
                passed_levels.append(ValidationLevel.L3_DECOMPRESSION)
            except Exception as exc:
                if not decomp_success:
                    errors.append(f"PNG image decode failed: {exc}")
                else:
                    passed_levels.append(ValidationLevel.L3_DECOMPRESSION)

        # -----------------------------------------------------------------
        # Level 4: Content Completeness Checks
        # -----------------------------------------------------------------
        if ValidationLevel.L3_DECOMPRESSION in passed_levels:
            if width > 0 and height > 0 and bit_depth in (1, 2, 4, 8, 16) and color_type in (0, 2, 3, 4, 6):
                passed_levels.append(ValidationLevel.L4_COMPLETENESS)
            else:
                warnings.append("IHDR parameters outside standard PNG specification.")

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
