"""Base Format Validator and Multi-Level Verification Hierarchy (v2.0.0-phase6).

Defines the six formal forensic validation tiers:
- L0: Signature recognition (magic byte verification).
- L1: Basic structural layout (headers, footers, segment bounds).
- L2: Structural parsing (chunks, objects, directory tables, cross-references).
- L3: Successful decompression or stream decoding (zlib, Huffman, DCT, audio frames).
- L4: Content integrity and completeness (valid dimensions, XML validation, metadata).
- L5: Byte-perfect ground-truth verification (STRICTLY reserved for controlled evaluation).
"""
from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


class ValidationLevel(IntEnum):
    """Multi-tiered forensic validation levels."""
    L0_SIGNATURE = 0      # Signature / magic byte presence
    L1_LAYOUT = 1         # Basic header/footer bounds and length consistency
    L2_STRUCTURE = 2      # Successful structural syntax parsing
    L3_DECOMPRESSION = 3  # Decompression or stream decoding
    L4_COMPLETENESS = 4   # Semantic content completeness and valid dimensions/metadata
    L5_GROUND_TRUTH = 5   # Exact byte-perfect reference match (evaluation only)


@dataclass
class FormatValidationResult:
    """Standardized validation verdict container for reconstructed files."""
    format_name: str
    highest_passed_level: ValidationLevel
    passed_levels: List[ValidationLevel]
    is_structurally_valid: bool       # True if at least L2 passed
    is_fully_renderable: bool         # True if L3 or L4 passed
    has_ground_truth_match: bool      # True if L5 passed (when reference provided)
    byte_length: int
    sha256_hash: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extracted_metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_name": self.format_name,
            "highest_passed_level": f"L{self.highest_passed_level.value}_{self.highest_passed_level.name}",
            "passed_levels": [f"L{lvl.value}" for lvl in self.passed_levels],
            "is_structurally_valid": self.is_structurally_valid,
            "is_fully_renderable": self.is_fully_renderable,
            "has_ground_truth_match": self.has_ground_truth_match,
            "byte_length": self.byte_length,
            "sha256_hash": self.sha256_hash,
            "errors": self.errors,
            "warnings": self.warnings,
            "extracted_metadata": self.extracted_metadata,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
        }


class BaseFormatValidator(abc.ABC):
    """Abstract base class for all independent forensic format validators."""

    def __init__(self, format_name: str):
        self.format_name = format_name

    @abc.abstractmethod
    def validate(
        self,
        raw_bytes: bytes,
        reference_hash: Optional[str] = None,
    ) -> FormatValidationResult:
        """Execute multi-tier validation on raw byte sequence."""
        raise NotImplementedError
