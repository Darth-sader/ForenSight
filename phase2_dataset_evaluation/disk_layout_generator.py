"""Physical Disk Layout and Carving Environment Generator (v2.0.0-phase2).

Simulates realistic storage media layouts with independent physical placement
and logical file order:
- Contiguous, partially fragmented, scattered, and multi-file interleaved layouts.
- Configurable fragmentation density and cluster run lengths.
- Controlled bit/byte corruption injection.
- Missing fragment drops (unallocated gaps / bad sectors).
- Realistic decoy fragment injection in unallocated slack.
- Explicit physical disk regions: VBR/Superblock, Allocation Table, Data Area, Slack Pool.
- Strict isolation: physical disk image & observable sector stream are decoupled
  from the ground-truth layout manifest.
"""
from __future__ import annotations

import copy
import hashlib
import random
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

FSIM_V2_MAGIC = b"FSIM_V20"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass
class DiskSector:
    sector_id: int
    physical_lba: int
    raw_bytes: bytes
    allocation_status: str  # "allocated", "unallocated", "slack", "system"
    region: str            # "vbr_superblock", "alloc_bitmap", "data_area", "slack_pool"
    is_decoy: bool = False

    def to_observable_dict(self, hex_bytes: bool = True) -> Dict[str, Any]:
        """Observable evidence accessible to carver without ground truth."""
        return {
            "fragment_id": f"sec_{self.sector_id:06d}",
            "observed_offset": self.physical_lba,
            "fragment_bytes_hex": self.raw_bytes.hex() if hex_bytes else self.raw_bytes,
            "fragment_length": len(self.raw_bytes),
            "allocation_status": self.allocation_status,
            "filesystem_region": self.region,
            "filesystem_metadata": {
                "sector_id": self.sector_id,
                "lba": self.physical_lba,
                "fs_magic": "FSIM_V2",
            },
            "acquisition_source": "simulated_disk_image",
        }


@dataclass
class SectorGroundTruth:
    sector_id: int
    physical_lba: int
    source_file_id: Optional[str]
    logical_offset: Optional[int]
    logical_sequence_index: Optional[int]
    original_fragment_id: Optional[str]
    is_corrupted: bool = False
    corruption_level: float = 0.0
    is_missing: bool = False
    is_decoy: bool = False
    decoy_type: Optional[str] = None
    original_file_hash: Optional[str] = None


@dataclass
class LayoutConfig:
    scenario_id: str
    block_size: int = 512
    interleave_factor: int = 0      # Foreign sectors inserted between file runs
    cluster_run_length: int = 4     # Consecutive sectors allocated per cluster run
    corruption_level: float = 0.0   # Fraction of payload bytes corrupted
    missing_ratio: float = 0.0      # Fraction of logical fragments dropped
    decoy_count: int = 0            # Number of decoy fragments injected
    noise_mode: str = "zero"        # "zero" | "random" | "pseudorandom"
    seed: int = 42


class PhysicalDiskLayoutGenerator:
    """Simulates realistic forensic disk images with decoupled physical and logical layouts."""

    def __init__(self, block_size: int = 512, superblock_sectors: int = 8, seed: int = 42):
        self.block_size = block_size
        self.superblock_sectors = superblock_sectors
        self.seed = seed

    def _pad_block(self, b: bytes) -> bytes:
        if len(b) >= self.block_size:
            return b[:self.block_size]
        return b.ljust(self.block_size, b"\x00")

    def _apply_corruption(self, buf: bytes, level: float, rng: random.Random) -> bytes:
        if level <= 0.0:
            return buf
        arr = bytearray(buf)
        n_corrupt = int(round(len(arr) * level))
        if n_corrupt > 0:
            positions = rng.sample(range(len(arr)), n_corrupt)
            noise = rng.randbytes(n_corrupt)
            for p, val in zip(positions, noise):
                arr[p] = val
        return bytes(arr)

    def generate_disk_image(
        self,
        files: Dict[str, bytes],
        config: LayoutConfig,
        decoy_payloads: Optional[List[bytes]] = None,
    ) -> Tuple[bytes, List[DiskSector], List[SectorGroundTruth]]:
        """Layout files across physical sectors and emit disk image, observable stream, and ground truth."""
        rng = random.Random(config.seed)
        decoy_payloads = decoy_payloads or []

        # 1. Fragment all input files into logical blocks
        file_logical_blocks: Dict[str, List[bytes]] = {}
        file_hashes: Dict[str, str] = {}
        for fname, payload in files.items():
            file_hashes[fname] = sha256_bytes(payload)
            blocks = []
            for off in range(0, len(payload), self.block_size):
                chunk = payload[off:off + self.block_size]
                blocks.append(self._pad_block(chunk))
            file_logical_blocks[fname] = blocks

        # 2. Determine file layout order and interleaving
        filenames = sorted(files.keys())
        rng.shuffle(filenames)

        # Plan physical sector allocations
        allocated_sectors: List[Tuple[Optional[str], Optional[int], bytes, bool, float, bool, Optional[str]]] = []
        # Tuple: (source_file, logical_index, block_bytes, is_corrupted, corrupt_level, is_decoy, decoy_type)

        if config.interleave_factor == 0:
            # Contiguous placement per file
            for fn in filenames:
                blocks = file_logical_blocks[fn]
                for seq_idx, blk in enumerate(blocks):
                    # Check missing fragment drop
                    if config.missing_ratio > 0 and rng.random() < config.missing_ratio:
                        continue  # Dropped from physical disk
                    # Check corruption
                    is_corr = False
                    corr_lvl = 0.0
                    out_blk = blk
                    if config.corruption_level > 0 and rng.random() < 0.5:
                        out_blk = self._apply_corruption(blk, config.corruption_level, rng)
                        is_corr = True
                        corr_lvl = config.corruption_level
                    allocated_sectors.append((fn, seq_idx, out_blk, is_corr, corr_lvl, False, None))
        else:
            # Multi-file interleaved placement
            # Round-robin allocation in runs of `cluster_run_length`
            file_pointers = {fn: 0 for fn in filenames}
            active_files = list(filenames)

            while active_files:
                for fn in list(active_files):
                    blocks = file_logical_blocks[fn]
                    ptr = file_pointers[fn]
                    run_len = min(config.cluster_run_length, len(blocks) - ptr)
                    for r in range(run_len):
                        seq_idx = ptr + r
                        blk = blocks[seq_idx]
                        if config.missing_ratio > 0 and rng.random() < config.missing_ratio:
                            continue
                        is_corr = False
                        corr_lvl = 0.0
                        out_blk = blk
                        if config.corruption_level > 0 and rng.random() < 0.5:
                            out_blk = self._apply_corruption(blk, config.corruption_level, rng)
                            is_corr = True
                            corr_lvl = config.corruption_level
                        allocated_sectors.append((fn, seq_idx, out_blk, is_corr, corr_lvl, False, None))

                    file_pointers[fn] += run_len
                    if file_pointers[fn] >= len(blocks):
                        active_files.remove(fn)

                    # Insert foreign slack / gap sectors according to interleave factor
                    for _ in range(config.interleave_factor):
                        gap_filler = b"\x00" * self.block_size if config.noise_mode == "zero" else rng.randbytes(self.block_size)
                        allocated_sectors.append((None, None, gap_filler, False, 0.0, False, None))

        # 3. Inject decoy fragments if configured
        if config.decoy_count > 0 and decoy_payloads:
            for d_idx in range(config.decoy_count):
                decoy_raw = self._pad_block(rng.choice(decoy_payloads))
                insert_pos = rng.randint(0, len(allocated_sectors))
                allocated_sectors.insert(
                    insert_pos,
                    (None, None, decoy_raw, False, 0.0, True, f"decoy_{d_idx}")
                )

        # 4. Construct complete disk image with system header sectors (VBR + Allocation table)
        sectors: List[DiskSector] = []
        ground_truths: List[SectorGroundTruth] = []

        total_data_sectors = len(allocated_sectors)
        total_sectors = self.superblock_sectors + total_data_sectors

        # Sector 0: Volume Boot Record / Superblock
        vbr_bytes = bytearray(self.block_size)
        vbr_bytes[:8] = FSIM_V2_MAGIC
        struct.pack_into("<I", vbr_bytes, 8, self.block_size)
        struct.pack_into("<I", vbr_bytes, 12, total_sectors)
        struct.pack_into("<I", vbr_bytes, 16, self.superblock_sectors)
        struct.pack_into("<I", vbr_bytes, 20, total_data_sectors)
        vbr_bytes = bytes(vbr_bytes)

        sectors.append(DiskSector(
            sector_id=0,
            physical_lba=0,
            raw_bytes=vbr_bytes,
            allocation_status="system",
            region="vbr_superblock",
        ))
        ground_truths.append(SectorGroundTruth(
            sector_id=0,
            physical_lba=0,
            source_file_id=None,
            logical_offset=None,
            logical_sequence_index=None,
            original_fragment_id=None,
        ))

        # Sectors 1 through superblock_sectors-1: Simulated allocation bitmap / reserved
        for s_idx in range(1, self.superblock_sectors):
            sys_bytes = bytearray(self.block_size)
            sectors.append(DiskSector(
                sector_id=s_idx,
                physical_lba=s_idx * self.block_size,
                raw_bytes=bytes(sys_bytes),
                allocation_status="system",
                region="alloc_bitmap",
            ))
            ground_truths.append(SectorGroundTruth(
                sector_id=s_idx,
                physical_lba=s_idx * self.block_size,
                source_file_id=None,
                logical_offset=None,
                logical_sequence_index=None,
                original_fragment_id=None,
            ))

        # Sectors superblock_sectors+: Data area sectors
        for i, item in enumerate(allocated_sectors):
            sec_id = self.superblock_sectors + i
            lba = sec_id * self.block_size
            fn, seq_idx, raw_b, is_corr, corr_lvl, is_decoy, decoy_type = item

            status = "unallocated" if (fn is None and not is_decoy) else "allocated"
            region = "data_area" if status == "allocated" else "slack_pool"

            sec = DiskSector(
                sector_id=sec_id,
                physical_lba=lba,
                raw_bytes=raw_b,
                allocation_status=status,
                region=region,
                is_decoy=is_decoy,
            )
            sectors.append(sec)

            gt = SectorGroundTruth(
                sector_id=sec_id,
                physical_lba=lba,
                source_file_id=fn,
                logical_offset=(seq_idx * self.block_size) if seq_idx is not None else None,
                logical_sequence_index=seq_idx,
                original_fragment_id=f"{fn}_frag_{seq_idx:04d}" if fn else None,
                is_corrupted=is_corr,
                corruption_level=corr_lvl,
                is_missing=False,
                is_decoy=is_decoy,
                decoy_type=decoy_type,
                original_file_hash=file_hashes.get(fn) if fn else None,
            )
            ground_truths.append(gt)

        # Assemble single contiguous disk image buffer
        raw_disk_image = b"".join(s.raw_bytes for s in sectors)
        return raw_disk_image, sectors, ground_truths
