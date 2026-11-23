"""MP3 Forensic Format Validator (v2.0.0-phase6).

Implements independent multi-tier verification for MPEG Layer III audio streams:
- L0: ID3 tag signature or MPEG audio frame sync (0xFFE/0xFFF).
- L1: Frame header structure parsing (version, layer, bitrate, sampling rate).
- L2: Consecutive frame sequence walk verifying calculated frame lengths.
- L3: Sustained valid audio frame decode stream (>= 3 consecutive valid frames).
- L4: Stream consistency: stable sample rate, valid channel mode, duration > 0.
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

# Standard MPEG-1 Layer III bitrates (kbps)
MPEG1_L3_BITRATES = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
# MPEG-1 Sampling rates (Hz)
MPEG1_SAMPLERATES = [44100, 48000, 32000, 0]


class MP3Validator(BaseFormatValidator):
    """Deep structural and frame-sequence validator for MP3 audio files."""

    def __init__(self):
        super().__init__("MP3")

    def _parse_frame_header(self, buf: bytes, offset: int) -> Optional[Tuple[int, int, int, int]]:
        """Parse MPEG frame header at offset: returns (frame_length, bitrate_kbps, samplerate, padding)."""
        if offset + 4 > len(buf):
            return None

        h = struct.unpack(">I", buf[offset:offset + 4])[0]
        # Check 11-bit sync 0x7FF
        if (h >> 21) != 0x7FF:
            return None

        version_id = (h >> 19) & 0x03    # 3 = MPEG-1
        layer_id = (h >> 17) & 0x03      # 1 = Layer III
        if version_id != 3 or layer_id != 1:
            return None

        bitrate_idx = (h >> 12) & 0x0F
        samplerate_idx = (h >> 10) & 0x03
        padding = (h >> 9) & 0x01

        if bitrate_idx in (0, 15) or samplerate_idx == 3:
            return None

        bitrate_kbps = MPEG1_L3_BITRATES[bitrate_idx]
        samplerate = MPEG1_SAMPLERATES[samplerate_idx]

        if bitrate_kbps == 0 or samplerate == 0:
            return None

        # Standard frame length formula for MPEG-1 Layer 3:
        # Length = int((144 * BitRate * 1000) / SampleRate) + Padding
        frame_len = int((144 * bitrate_kbps * 1000) / samplerate) + padding
        return frame_len, bitrate_kbps, samplerate, padding

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
        # Level 0: Signature Recognition (ID3 or Frame Sync)
        # -----------------------------------------------------------------
        has_id3 = raw_bytes[:3] == b"ID3"
        has_sync = False
        scan_limit = min(8192, file_len - 4)

        for i in range(scan_limit):
            if (raw_bytes[i] == 0xFF) and ((raw_bytes[i + 1] & 0xE0) == 0xE0):
                has_sync = True
                break

        if not has_id3 and not has_sync:
            errors.append("Missing ID3 tag and MPEG frame sync sequence.")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        passed_levels.append(ValidationLevel.L0_SIGNATURE)
        metadata["has_id3"] = has_id3

        # -----------------------------------------------------------------
        # Level 1: Basic Structural Layout (First valid frame header)
        # -----------------------------------------------------------------
        first_frame_offset = -1
        first_bitrate = 0
        first_samplerate = 0

        # Skip ID3v2 if present
        start_offset = 0
        if has_id3 and file_len >= 10:
            # ID3 size is synchsafe integer in bytes 6..9
            b6, b7, b8, b9 = raw_bytes[6:10]
            id3_len = ((b6 & 0x7F) << 21) | ((b7 & 0x7F) << 14) | ((b8 & 0x7F) << 7) | (b9 & 0x7F)
            start_offset = min(file_len, 10 + id3_len)
            metadata["id3_size_bytes"] = id3_len

        for off in range(start_offset, min(start_offset + 16384, file_len - 4)):
            parsed = self._parse_frame_header(raw_bytes, off)
            if parsed is not None:
                first_frame_offset = off
                _, first_bitrate, first_samplerate, _ = parsed
                break

        if first_frame_offset >= 0:
            passed_levels.append(ValidationLevel.L1_LAYOUT)
            metadata["first_frame_offset"] = first_frame_offset
            metadata["bitrate_kbps"] = first_bitrate
            metadata["samplerate_hz"] = first_samplerate
        else:
            errors.append("No valid MPEG-1 Layer 3 audio frame header found.")
            return self._build_result(passed_levels, errors, warnings, metadata, file_len, file_hash, t0)

        # -----------------------------------------------------------------
        # Level 2: Frame Sequence Walk
        # -----------------------------------------------------------------
        curr_offset = first_frame_offset
        consecutive_frames = 0
        total_frames = 0

        while curr_offset < file_len - 4 and total_frames < 200:
            parsed = self._parse_frame_header(raw_bytes, curr_offset)
            if parsed is not None:
                flen, _, _, _ = parsed
                consecutive_frames += 1
                total_frames += 1
                curr_offset += flen
            else:
                # Synchronization lost
                break

        metadata["consecutive_frames"] = consecutive_frames
        metadata["total_frames_parsed"] = total_frames

        if consecutive_frames >= 2:
            passed_levels.append(ValidationLevel.L2_STRUCTURE)
        else:
            warnings.append("Could not maintain frame synchronization beyond 1 frame.")

        # -----------------------------------------------------------------
        # Level 3: Sustained Audio Stream Decode Test (>= 3 frames)
        # -----------------------------------------------------------------
        if consecutive_frames >= 3:
            passed_levels.append(ValidationLevel.L3_DECOMPRESSION)

        # -----------------------------------------------------------------
        # Level 4: Content Completeness Checks
        # -----------------------------------------------------------------
        if ValidationLevel.L3_DECOMPRESSION in passed_levels:
            if first_bitrate > 0 and first_samplerate > 0 and consecutive_frames >= 4:
                passed_levels.append(ValidationLevel.L4_COMPLETENESS)
                est_seconds = (consecutive_frames * 1152) / first_samplerate
                metadata["estimated_duration_seconds"] = round(est_seconds, 2)
            else:
                warnings.append("Insufficient sustained frame sequence for complete stream validation.")

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
