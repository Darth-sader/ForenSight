"""Versioned Forensic Fragment Schema (v2.0.0-leakage_free).

Enforces strict separation between ground-truth labels and observable
forensic evidence to guarantee that zero logical positions or ground-truth
file ordering leak into inference pipelines.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union


class SchemaLeakageError(ValueError):
    """Raised when ground-truth fields are detected in inference inputs."""
    pass


# Strict registry of ground-truth fields that MUST NEVER appear in inference inputs
GROUND_TRUTH_ONLY_FIELDS: Set[str] = {
    "source_file_id",
    "source_file",
    "logical_offset",
    "offset_in_source",
    "logical_sequence_index",
    "original_position",
    "original_fragment_id",
    "original_bytes",
    "is_corrupted",
    "is_missing",
    "original_file_hash",
    "similar_group",
    "split",
    "variant",
}

# Permitted observable fields accessible during real forensic acquisition/carving
OBSERVABLE_FIELDS: Set[str] = {
    "fragment_id",
    "observed_offset",
    "fragment_bytes",
    "fragment_length",
    "allocation_status",
    "filesystem_metadata",
    "filesystem_region",
    "acquisition_source",
}


@dataclass(frozen=True)
class ObservableFragment:
    """Forensic evidence container containing strictly observable physical properties."""
    fragment_id: str
    observed_offset: int
    fragment_bytes: bytes
    fragment_length: int
    allocation_status: str = "unallocated"  # "allocated", "unallocated", "slack", "raw"
    filesystem_metadata: Dict[str, Any] = field(default_factory=dict)
    filesystem_region: str = "data_area"    # "data_area", "unallocated_pool", "carved_stream"
    acquisition_source: str = "disk_image"  # "disk_image", "memory_dump", "file_carver"

    def __post_init__(self):
        if self.fragment_length != len(self.fragment_bytes):
            raise ValueError(
                f"fragment_length ({self.fragment_length}) does not match bytes length ({len(self.fragment_bytes)})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "observed_offset": self.observed_offset,
            "fragment_bytes": self.fragment_bytes,
            "fragment_length": self.fragment_length,
            "allocation_status": self.allocation_status,
            "filesystem_metadata": copy.deepcopy(self.filesystem_metadata),
            "filesystem_region": self.filesystem_region,
            "acquisition_source": self.acquisition_source,
        }


@dataclass
class GroundTruthFragment:
    """Complete experimental entity combining ground-truth truth labels and observable evidence."""
    # Observable evidence
    observable: ObservableFragment

    # Ground-truth-only fields (used solely for label generation, never inference)
    source_file_id: str
    logical_offset: int
    logical_sequence_index: int
    original_fragment_id: str
    original_bytes: bytes
    is_corrupted: bool = False
    is_missing: bool = False
    original_file_hash: Optional[str] = None
    similar_group: Optional[str] = None
    split: str = "train"  # "train", "val", "cal", "test"

    def to_observable(self) -> ObservableFragment:
        """Strip all ground truth and return pure observable fragment."""
        return copy.deepcopy(self.observable)


def validate_observable_record(record: Union[Dict[str, Any], ObservableFragment]) -> None:
    """Verify that a record contains zero ground-truth fields.
    
    Raises:
        SchemaLeakageError: If any forbidden ground-truth attribute is found.
    """
    if isinstance(record, ObservableFragment):
        return  # Type-level guarantee

    if not isinstance(record, dict):
        raise TypeError(f"Expected dict or ObservableFragment, got {type(record)}")

    leaked = [k for k in record.keys() if k in GROUND_TRUTH_ONLY_FIELDS]
    if leaked:
        raise SchemaLeakageError(
            f"CRITICAL SCHEMA LEAKAGE DETECTED: Ground-truth fields {leaked} found in inference record!"
        )


def sanitize_to_observable_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    """Given a raw or hybrid dictionary, strip all ground-truth fields and return a clean observable dict."""
    clean = {}
    for k, v in record.items():
        if k in OBSERVABLE_FIELDS:
            clean[k] = copy.deepcopy(v) if isinstance(v, (dict, list)) else v
    return clean
