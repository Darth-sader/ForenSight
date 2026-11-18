"""Fragment extraction from disk images (doc §4, §8, §39).

Reads a forensic disk image and yields fixed-size fragments with their
filesystem context.  Works with ForenSight's FSIM images, arbitrary raw
images, and (optionally) real filesystem images via pytsk3.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from common import Fragment, set_seeds
from config import FRAGMENT_SIZE
from extraction.filesystem_parser import DiskImageParser, try_pytsk3_listing


def extract_fragments(
    image_path: str | Path,
    fragment_size: int = FRAGMENT_SIZE,
    fill_unallocated: bool = True,
    seed: int = 42,
) -> Tuple[List[Fragment], np.ndarray, Dict[str, object]]:
    """Extract all blocks of ``image_path`` as fragments.

    Returns ``(fragments, data_matrix, image_summary)`` where ``data_matrix``
    is an ``(n, fragment_size)`` uint8 array in fragment order.
    """
    set_seeds(seed)
    parser = DiskImageParser(image_path, block_size=fragment_size)
    n_blocks = parser.block_count()
    data = np.empty((n_blocks, fragment_size), dtype=np.uint8)
    fragments: List[Fragment] = []
    tsk_note = None

    # Real-filesystem images: try Sleuth Kit metadata first (optional).
    if parser.info.fs_type == "RAW":
        tsk = try_pytsk3_listing(image_path)
        if tsk is not None:
            tsk_note = tsk

    for i in range(n_blocks):
        block = parser.read_block(i)
        data[i] = np.frombuffer(block, dtype=np.uint8)
        status = parser.block_allocation_status(i)
        ctx = {
            "fs_type": parser.info.fs_type,
            "volume_id": parser.info.volume_id,
            "cluster": i,
            "neighbors": [i - 1, i + 1],
            "region": "system" if status == "system" else "data",
        }
        fragments.append(Fragment(
            fragment_id=f"F{i:05d}",
            fragment_size=fragment_size,
            physical_offset=i * fragment_size,
            cluster=i,
            allocation_status=status,
            filesystem_context=ctx,
        ))
    summary = parser.summary()
    if tsk_note:
        summary["pytsk3"] = {k: v for k, v in tsk_note.items() if k != "files"}
        summary["pytsk3_file_count"] = len(tsk_note.get("files", []))
    parser.close()
    return fragments, data, summary


def fragments_from_bytes(blocks: List[bytes], start_offset: int = 0,
                         allocation: Optional[List[str]] = None,
                         volume_id: str = "") -> Tuple[List[Fragment], np.ndarray]:
    """Build fragment records from in-memory blocks (corpus generation path)."""
    n = len(blocks)
    data = np.empty((n, len(blocks[0]) if blocks else 0), dtype=np.uint8)
    fragments: List[Fragment] = []
    for i, b in enumerate(blocks):
        data[i] = np.frombuffer(b, dtype=np.uint8)
        fragments.append(Fragment(
            fragment_id=f"F{i:05d}",
            fragment_size=len(b),
            physical_offset=start_offset + i * len(b),
            cluster=i,
            allocation_status=(allocation[i] if allocation else "allocated"),
            filesystem_context={"fs_type": "FSIM", "volume_id": volume_id,
                                "cluster": i, "neighbors": [i - 1, i + 1],
                                "region": "data"},
        ))
    return fragments, data


def corrupt_fragment(data: np.ndarray, pct: float, rng: random.Random) -> np.ndarray:
    """Corrupt ``pct`` fraction of bytes in-place copy (doc scenario D)."""
    if pct <= 0:
        return data
    out = data.copy()
    n_bytes = int(round(out.size * pct))
    if n_bytes == 0:
        return out
    idx = rng.sample(range(out.size), min(n_bytes, out.size))
    noise = np.frombuffer(rng.randbytes(len(idx)), dtype=np.uint8)
    out[idx] = noise
    return out
