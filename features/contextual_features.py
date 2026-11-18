"""Contextual features (doc §9A): physical offset, filesystem cluster,
allocation status and neighbouring blocks."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_contextual_features(meta: Dict[str, object],
                                max_offset: float) -> Dict[str, float]:
    ctx = meta.get("filesystem_context") or {}
    alloc = str(meta.get("allocation_status", "raw"))
    neighbors = ctx.get("neighbors") or [meta.get("cluster", -1)] * 2
    return {
        "alloc_allocated": 1.0 if alloc == "allocated" else 0.0,
        "alloc_unallocated": 1.0 if alloc == "unallocated" else 0.0,
        "alloc_system": 1.0 if alloc in ("system", "raw") else 0.0,
        "offset_norm": float(meta.get("physical_offset", 0)) / max(1.0, max_offset),
        "cluster_norm": float(meta.get("cluster", 0)) / max(1.0, max_offset),
        "prev_neighbor_present": 1.0 if neighbors[0] is not None and neighbors[0] >= 0 else 0.0,
        "next_neighbor_present": 1.0 if len(neighbors) > 1 and neighbors[1] is not None else 0.0,
        "offset_mod_1000": (float(meta.get("physical_offset", 0)) % 1000) / 1000.0,
        "fs_fsim": 1.0 if str(ctx.get("fs_type", "")).upper() == "FSIM" else 0.0,
        "fs_raw": 1.0 if str(ctx.get("fs_type", "")).upper() in ("RAW", "") else 0.0,
    }


def compute_contextual_matrix(metas: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    max_offset = max([float(m.get("physical_offset") or 0) for m in metas] or [1.0]) or 1.0
    names = SINGLE_CONTEXTUAL_NAMES
    out = {k: np.zeros(len(metas)) for k in names}
    for i, m in enumerate(metas):
        feats = compute_contextual_features(m, max_offset)
        for k in names:
            out[k][i] = feats[k]
    return out


def single_contextual_names() -> List[str]:
    return ["alloc_allocated", "alloc_unallocated", "alloc_system", "offset_norm",
            "cluster_norm", "prev_neighbor_present", "next_neighbor_present",
            "offset_mod_1000", "fs_fsim", "fs_raw"]


SINGLE_CONTEXTUAL_NAMES = single_contextual_names()
