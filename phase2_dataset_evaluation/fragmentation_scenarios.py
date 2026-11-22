"""Forensic Fragmentation Scenario Matrix and Generator (v2.0.0-phase2).

Implements a systematic experimental matrix across 5 forensic dimensions:
1. Fragmentation Degree: Contiguous, Low (I=1), Medium (I=3), High (I=6), Extreme (I=12)
2. Interleaving Concurrency: 2 files, 5 files, 10 files, 15 files
3. Payload Corruption: 0%, 1%, 5%, 10%, 20%
4. Missing Evidence: 0%, 1%, 5%, 10%, 20%
5. Decoys & Noise: None, Low (5%), Medium (15%), High (30%)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phase2_dataset_evaluation.disk_layout_generator import (
    LayoutConfig,
    PhysicalDiskLayoutGenerator,
    SectorGroundTruth,
    DiskSector,
    sha256_bytes,
)

# Standard experimental scenarios covering all forensic challenge dimensions
SCENARIO_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "SCEN_A_CONTIG_CLEAN": {
        "description": "Baseline: Contiguous placement, clean payloads, 0% corruption, 0% missing.",
        "interleave_factor": 0,
        "cluster_run_length": 64,
        "corruption_level": 0.0,
        "missing_ratio": 0.0,
        "decoy_count": 0,
        "noise_mode": "zero",
        "file_subset_size": 10,
    },
    "SCEN_B_LOW_FRAG_2F": {
        "description": "Low fragmentation: 2-file interleaving, interleave factor 1, run length 8.",
        "interleave_factor": 1,
        "cluster_run_length": 8,
        "corruption_level": 0.0,
        "missing_ratio": 0.0,
        "decoy_count": 0,
        "noise_mode": "zero",
        "file_subset_size": 2,
    },
    "SCEN_C_MED_FRAG_5F": {
        "description": "Medium fragmentation: 5-file interleaving, interleave factor 3, run length 4.",
        "interleave_factor": 3,
        "cluster_run_length": 4,
        "corruption_level": 0.0,
        "missing_ratio": 0.0,
        "decoy_count": 0,
        "noise_mode": "zero",
        "file_subset_size": 5,
    },
    "SCEN_D_HIGH_FRAG_10F": {
        "description": "High fragmentation: 10-file interleaving, interleave factor 6, run length 2.",
        "interleave_factor": 6,
        "cluster_run_length": 2,
        "corruption_level": 0.0,
        "missing_ratio": 0.0,
        "decoy_count": 0,
        "noise_mode": "random",
        "file_subset_size": 10,
    },
    "SCEN_E_EXTREME_FRAG_15F": {
        "description": "Extreme fragmentation: 15-file interleaving, interleave factor 12, run length 1.",
        "interleave_factor": 12,
        "cluster_run_length": 1,
        "corruption_level": 0.0,
        "missing_ratio": 0.0,
        "decoy_count": 0,
        "noise_mode": "random",
        "file_subset_size": 15,
    },
    "SCEN_F_CORRUPT_C05": {
        "description": "Corruption tolerance: 5-file interleave with 5% byte-level corruption.",
        "interleave_factor": 2,
        "cluster_run_length": 4,
        "corruption_level": 0.05,
        "missing_ratio": 0.0,
        "decoy_count": 0,
        "noise_mode": "zero",
        "file_subset_size": 5,
    },
    "SCEN_G_CORRUPT_C20": {
        "description": "Severe corruption tolerance: 5-file interleave with 20% byte-level corruption.",
        "interleave_factor": 2,
        "cluster_run_length": 4,
        "corruption_level": 0.20,
        "missing_ratio": 0.0,
        "decoy_count": 0,
        "noise_mode": "zero",
        "file_subset_size": 5,
    },
    "SCEN_H_MISSING_M05": {
        "description": "Missing fragments: 5-file interleave with 5% dropped blocks.",
        "interleave_factor": 2,
        "cluster_run_length": 4,
        "corruption_level": 0.0,
        "missing_ratio": 0.05,
        "decoy_count": 0,
        "noise_mode": "zero",
        "file_subset_size": 5,
    },
    "SCEN_I_MISSING_M20": {
        "description": "Severe missing fragments: 5-file interleave with 20% dropped blocks.",
        "interleave_factor": 2,
        "cluster_run_length": 4,
        "corruption_level": 0.0,
        "missing_ratio": 0.20,
        "decoy_count": 0,
        "noise_mode": "zero",
        "file_subset_size": 5,
    },
    "SCEN_J_DECOY_D15": {
        "description": "Decoy resistance: 5-file interleave with 15 injected decoy fragments.",
        "interleave_factor": 2,
        "cluster_run_length": 4,
        "corruption_level": 0.0,
        "missing_ratio": 0.0,
        "decoy_count": 15,
        "noise_mode": "random",
        "file_subset_size": 5,
    },
    "SCEN_K_ADVERSARIAL_STORM": {
        "description": "Stress test: 10-file interleaving, 10% corruption, 10% missing, 20 decoys.",
        "interleave_factor": 5,
        "cluster_run_length": 2,
        "corruption_level": 0.10,
        "missing_ratio": 0.10,
        "decoy_count": 20,
        "noise_mode": "random",
        "file_subset_size": 10,
    },
}


class ScenarioRunner:
    """Instantiates disk images and manifests for fragmentation scenarios."""

    def __init__(self, block_size: int = 512, seed: int = 42):
        self.block_size = block_size
        self.seed = seed
        self.generator = PhysicalDiskLayoutGenerator(block_size=block_size, seed=seed)

    def execute_scenario(
        self,
        scenario_id: str,
        files_corpus: Dict[str, bytes],
        decoy_payloads: Optional[List[bytes]] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute a scenario, save artifacts, and return observable metadata and ground truth."""
        if scenario_id not in SCENARIO_DEFINITIONS:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")

        spec = SCENARIO_DEFINITIONS[scenario_id]

        # Select file subset deterministically
        import random
        rng = random.Random(self.seed + hash(scenario_id) % 10000)
        sorted_files = sorted(files_corpus.keys())
        k = min(spec["file_subset_size"], len(sorted_files))
        selected_files = rng.sample(sorted_files, k)
        subset_payloads = {fn: files_corpus[fn] for fn in selected_files}

        cfg = LayoutConfig(
            scenario_id=scenario_id,
            block_size=self.block_size,
            interleave_factor=spec["interleave_factor"],
            cluster_run_length=spec["cluster_run_length"],
            corruption_level=spec["corruption_level"],
            missing_ratio=spec["missing_ratio"],
            decoy_count=spec["decoy_count"],
            noise_mode=spec["noise_mode"],
            seed=self.seed,
        )

        disk_image, sectors, ground_truths = self.generator.generate_disk_image(
            files=subset_payloads,
            config=cfg,
            decoy_payloads=decoy_payloads,
        )

        # Build isolated Ground-Truth Manifest adhering strictly to ground_truth_schema_v2.json
        files_gt: Dict[str, Any] = {}
        for fn in selected_files:
            payload = subset_payloads[fn]
            file_secs = [gt for gt in ground_truths if gt.source_file_id == fn]
            file_secs.sort(key=lambda s: s.logical_sequence_index)
            corr_indices = [s.logical_sequence_index for s in file_secs if s.is_corrupted]

            total_fragments = (len(payload) + self.block_size - 1) // self.block_size
            present_indices = {s.logical_sequence_index for s in file_secs}
            missing_indices = [i for i in range(total_fragments) if i not in present_indices]

            # Collect physical LBAs corresponding to each logical index
            lba_map = {s.logical_sequence_index: s.physical_lba for s in file_secs}
            physical_locations = [lba_map.get(i, None) for i in range(total_fragments)]

            # Fragment hashes
            frag_hashes = []
            for off in range(0, len(payload), self.block_size):
                chunk = payload[off:off + self.block_size]
                if len(chunk) < self.block_size:
                    chunk = chunk.ljust(self.block_size, b"\x00")
                frag_hashes.append(sha256_bytes(chunk))

            files_gt[fn] = {
                "source_file_id": fn,
                "original_file_hash": sha256_bytes(payload),
                "file_size_bytes": len(payload),
                "total_fragments": total_fragments,
                "file_type": fn.split(".")[-1].upper(),
                "logical_byte_offsets": [i * self.block_size for i in range(total_fragments)],
                "logical_fragment_sequence": list(range(total_fragments)),
                "fragment_hashes": frag_hashes,
                "observed_physical_locations": physical_locations,
                "corruption_information": {
                    "corrupted_fragment_count": len(corr_indices),
                    "corrupted_indices": corr_indices,
                },
                "missing_fragment_information": {
                    "missing_fragment_count": len(missing_indices),
                    "missing_indices": missing_indices,
                },
            }

        sectors_gt = [asdict(gt) for gt in ground_truths]

        gt_manifest = {
            "schema_version": "2.0.0-ground_truth",
            "scenario_id": scenario_id,
            "source_provenance": {
                "generator": "PhysicalDiskLayoutGenerator",
                "seed": self.seed,
                "corpus_version": "2.0.0-phase2",
                "interleave_factor": spec["interleave_factor"],
                "corruption_level": spec["corruption_level"],
                "missing_ratio": spec["missing_ratio"],
                "decoy_count": spec["decoy_count"],
            },
            "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_files": len(selected_files),
            "total_sectors": len(sectors),
            "files": files_gt,
            "sectors": sectors_gt,
        }

        # Observable evidence manifest (observable by carver without ground-truth fields)
        observable_evidence = [sec.to_observable_dict(hex_bytes=True) for sec in sectors if sec.allocation_status != "system"]

        if output_dir:
            out_p = Path(output_dir) / scenario_id
            out_p.mkdir(parents=True, exist_ok=True)
            # Write raw disk image
            (out_p / f"{scenario_id}.raw").write_bytes(disk_image)
            # Write observable evidence
            with open(out_p / "observable_evidence.json", "w", encoding="utf-8") as f:
                json.dump(observable_evidence, f, indent=2)
            # Write isolated ground truth
            with open(out_p / "ground_truth_manifest.json", "w", encoding="utf-8") as f:
                json.dump(gt_manifest, f, indent=2)

        return {
            "scenario_id": scenario_id,
            "disk_image_bytes": len(disk_image),
            "total_sectors": len(sectors),
            "observable_sectors": len(observable_evidence),
            "files_count": len(selected_files),
            "ground_truth_manifest": gt_manifest,
            "observable_evidence": observable_evidence,
        }
