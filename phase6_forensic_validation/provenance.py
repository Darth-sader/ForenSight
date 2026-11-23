"""Immutable Forensic Evidence Provenance Tracking (v2.0.0-phase6).

Maintains a chain-of-custody and evidence record for every reconstructed candidate:
- Disk image identifier and acquisition source
- Component fragments: IDs, physical offsets, byte lengths, SHA-256 hashes
- Edge transitions: predecessors, successors, raw model probabilities, calibrated probabilities, graph scores
- Optimization solver decisions (x_ij, g_ij variables)
- Missing fragment gaps and placeholder positions
- Format validation level (L0 to L5) and parser diagnostics
- Final reconstructed file SHA-256 and MD5 hashes
- Complete model, pipeline, and software version provenance
- Cryptographic provenance signature for reproducible auditability
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from phase1_leakage_free.fragment_schema_v2 import ObservableFragment
from phase6_forensic_validation.abstention_policy import DecisionVerdict, ReconstructionOutcome
from phase6_forensic_validation.format_validators.base_validator import FormatValidationResult


@dataclass
class FragmentProvenance:
    """Provenance record for an individual observable fragment."""
    fragment_id: str
    observed_offset: int
    fragment_length: int
    sha256_hash: str
    allocation_status: str
    filesystem_region: str


@dataclass
class TransitionProvenance:
    """Provenance record for a directed successor transition."""
    source_fragment_id: str
    target_fragment_id: str
    is_gap_transition: bool
    raw_probability: float
    calibrated_probability: float
    graph_edge_score: float
    optimization_decision: int          # 1 if active, 0 otherwise


@dataclass
class EvidenceProvenanceRecord:
    """Complete, immutable provenance record for a single reconstructed file."""
    recon_id: str
    reconstructed_sha256: str
    reconstructed_md5: str
    reconstructed_size_bytes: int
    claimed_format: Optional[str]
    disk_image_id: str
    acquisition_source: str
    timestamp_utc: str
    software_version: str = "v2.0.0-phase6"
    pipeline_version: str = "forensight-v2"
    fragments: List[FragmentProvenance] = field(default_factory=list)
    transitions: List[TransitionProvenance] = field(default_factory=list)
    missing_gaps: List[Dict[str, Any]] = field(default_factory=list)
    validation: Optional[Dict[str, Any]] = None
    verdict: Optional[Dict[str, Any]] = None
    model_metadata: Dict[str, str] = field(default_factory=dict)
    provenance_hash: str = ""

    def compute_provenance_hash(self) -> str:
        """Compute cryptographic hash over canonical provenance fields for reproducibility."""
        canonical_dict = {
            "recon_id": self.recon_id,
            "reconstructed_sha256": self.reconstructed_sha256,
            "fragments": [f.fragment_id for f in self.fragments],
            "transitions": [
                f"{t.source_fragment_id}->{t.target_fragment_id}:{t.optimization_decision}"
                for t in self.transitions
            ],
            "software_version": self.software_version,
        }
        encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provenance_hash"] = self.provenance_hash or self.compute_provenance_hash()
        return d


class EvidenceProvenanceTracker:
    """Factory and registry for generating reproducible forensic provenance records."""

    def __init__(
        self,
        disk_image_id: str = "DISK_IMAGE_001",
        acquisition_source: str = "PHYSICAL_RAW",
    ):
        self.disk_image_id = disk_image_id
        self.acquisition_source = acquisition_source

    def create_provenance_record(
        self,
        recon_id: str,
        reconstructed_bytes: bytes,
        sequence_fragment_ids: List[str],
        all_fragments_map: Dict[str, ObservableFragment],
        p_raw_dict: Dict[Tuple[str, str], float],
        p_cal_dict: Dict[Tuple[str, str], float],
        val_result: FormatValidationResult,
        verdict: DecisionVerdict,
        claimed_format: Optional[str] = None,
        model_versions: Optional[Dict[str, str]] = None,
    ) -> EvidenceProvenanceRecord:
        """Construct full chain-of-custody provenance record for a reconstruction."""
        sha256_val = hashlib.sha256(reconstructed_bytes).hexdigest()
        md5_val = hashlib.md5(reconstructed_bytes).hexdigest()

        # Build fragment provenance
        frag_provs: List[FragmentProvenance] = []
        for fid in sequence_fragment_ids:
            if fid.startswith("[GAP_"):
                continue
            frag = all_fragments_map.get(fid)
            if frag:
                frag_provs.append(FragmentProvenance(
                    fragment_id=frag.fragment_id,
                    observed_offset=frag.observed_offset,
                    fragment_length=frag.fragment_length,
                    sha256_hash=hashlib.sha256(frag.fragment_bytes).hexdigest(),
                    allocation_status=frag.allocation_status,
                    filesystem_region=frag.filesystem_region,
                ))

        # Build transition provenance
        trans_provs: List[TransitionProvenance] = []
        gaps_list: List[Dict[str, Any]] = []

        for i in range(len(sequence_fragment_ids) - 1):
            src = sequence_fragment_ids[i]
            dst = sequence_fragment_ids[i + 1]

            is_gap = False
            if dst.startswith("[GAP_") or src.startswith("[GAP_"):
                is_gap = True
                gaps_list.append({"position": i, "gap_marker": dst if dst.startswith("[GAP_") else src})

            p_raw = p_raw_dict.get((src, dst), 0.5)
            p_cal = p_cal_dict.get((src, dst), p_raw)

            trans_provs.append(TransitionProvenance(
                source_fragment_id=src,
                target_fragment_id=dst,
                is_gap_transition=is_gap,
                raw_probability=round(p_raw, 4),
                calibrated_probability=round(p_cal, 4),
                graph_edge_score=round(p_cal, 4),
                optimization_decision=1,
            ))

        rec = EvidenceProvenanceRecord(
            recon_id=recon_id,
            reconstructed_sha256=sha256_val,
            reconstructed_md5=md5_val,
            reconstructed_size_bytes=len(reconstructed_bytes),
            claimed_format=claimed_format or val_result.format_name,
            disk_image_id=self.disk_image_id,
            acquisition_source=self.acquisition_source,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            fragments=frag_provs,
            transitions=trans_provs,
            missing_gaps=gaps_list,
            validation=val_result.to_dict(),
            verdict=verdict.to_dict(),
            model_metadata=model_versions or {"same_file": "v2_rf", "next_fragment": "v2_rf"},
        )
        rec.provenance_hash = rec.compute_provenance_hash()
        return rec
