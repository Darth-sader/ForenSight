"""Pairwise feature engineering (doc §10): for fragments A and B compute
entropy difference, histogram distance, n-gram similarity, boundary/prefix-
suffix compatibility, structural and format compatibility, physical and
filesystem distances and context compatibility.

Every column is tagged with a feature family so the ablation studies (doc §28)
can build cumulative masks: byte -> +statistical -> +structural -> +boundary
-> +context.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from features.statistical_features import NGRAM_DIM

# (name, family) for every pair column, assembled by build_pair_names()
PAIR_NAMES: List[Tuple[str, str]] = []


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def boundary_byte_compatibility(a_bytes: np.ndarray, b_bytes: np.ndarray,
                                a_struct: Dict[str, float], b_struct: Dict[str, float]) -> Dict[str, float]:
    """Format-aware boundary checks plus generic byte continuity (doc §10)."""
    la, fb = int(a_bytes[-1]), int(b_bytes[0])
    out: Dict[str, float] = {}

    # generic continuity: boundary bigram plausible given each fragment's byte stats
    pa = float((a_bytes == la).mean())
    pb = float((b_bytes == fb).mean())
    out["boundary_byte_continuity"] = 1.0 - min(1.0, abs(pa - pb))

    # literal suffix/prefix overlap (doc: prefix/suffix compatibility)
    k = min(8, len(a_bytes), len(b_bytes))
    out["prefix_suffix_overlap"] = float(np.mean(a_bytes[-k:] == b_bytes[:k])) if k else 0.0

    rule = 0.5  # neutral when no rule applies
    if a_struct["header_JPEG"] > 0 or a_struct["internal_JPEG"] > 0 or la == 0xFF:
        if la == 0xFF and (0xD0 <= fb <= 0xD9 or fb in (0x01, 0xC0, 0xC1, 0xC2, 0xC4, 0xDA, 0xDB, 0xE0, 0xE1, 0xFE)):
            rule = 1.0
        elif la != 0xFF and fb != 0xFF:
            rule = 0.8
        else:
            rule = 0.1
    if a_struct["internal_PDF"] > 0 and a_bytes[-6:].tobytes().rstrip().endswith(b"endobj"):
        rule = max(rule, 1.0 if fb in b"0123456789<%" else 0.3)
    if a_struct["header_MP3"] > 0 or a_struct["internal_MP3"] > 0:
        if la == 0xFF and fb in (0xFB, 0xFA, 0xF3, 0xF2):
            rule = max(rule, 1.0)
    if a_struct["footer_PNG"] >= 1.0 or a_struct["footer_PDF"] >= 1.0 or a_struct["footer_ZIP"] >= 1.0:
        rule = min(rule, 0.2)  # B after a complete file is suspicious
        if b_struct["header_" + ("PNG" if a_struct["footer_PNG"] >= 1 else
                                 "PDF" if a_struct["footer_PDF"] >= 1 else "ZIP")] >= 0.8:
            rule = 0.0  # clearly a new file
    if b_struct["is_header_fragment"] >= 1.0 and a_struct["is_header_fragment"] < 1.0:
        rule = min(rule, 0.15)  # B starts a new file
    out["boundary_format_rule"] = rule

    out["boundary_incomplete_marker"] = 1.0 if (la == 0xFF or
                                                (a_bytes[-4:] == b"\x00\x00\x00\x01").all()) else 0.0
    out["boundary_frame_alignment"] = 1.0 if (la in (0xFB, 0xFA, 0xF3, 0xF2) and fb == 0xFF) else 0.0
    out["boundary_generic_score"] = float(np.clip(
        0.5 * rule + 0.3 * out["boundary_byte_continuity"] + 0.2 * (1.0 - out["prefix_suffix_overlap"] * 0),
        0.0, 1.0))
    return out


def structural_pair_features(a_struct: Dict[str, float], b_struct: Dict[str, float]) -> Dict[str, float]:
    fam_a = {t: (a_struct[f"header_{t}"] >= 0.8 or a_struct[f"internal_{t}"] > 0)
             for t in ("JPEG", "PNG", "PDF", "ZIP", "DOCX", "XLSX", "MP3", "MP4")}
    fam_b = {t: (b_struct[f"header_{t}"] >= 0.8 or b_struct[f"internal_{t}"] > 0)
             for t in fam_a}
    shared = sum(1.0 for t in fam_a if fam_a[t] and fam_b[t])
    claimed_a = sum(1.0 for v in fam_a.values() if v) or 1.0
    same_format = shared / claimed_a
    office = ("ZIP", "DOCX", "XLSX")
    fam_norm_a = {("office" if t in office else t): v for t, v in fam_a.items() if v}
    fam_norm_b = {("office" if t in office else t): v for t, v in fam_b.items() if v}
    same_family = 1.0 if set(fam_norm_a) & set(fam_norm_b) else 0.0
    return {
        "struct_same_format_detected": float(np.clip(same_format, 0.0, 1.0)),
        "struct_same_format_family": same_family,
        "struct_header_then_body": 1.0 if (a_struct["is_header_fragment"] >= 1.0
                                           and b_struct["is_header_fragment"] < 1.0) else 0.0,
        "struct_footer_then_any": 1.0 if (a_struct["is_footer_fragment"] >= 1.0) else 0.0,
        "struct_header_in_b": 1.0 if b_struct["is_header_fragment"] >= 1.0 else 0.0,
        "struct_conflict_sum": float(a_struct["multi_format_conflict"] + b_struct["multi_format_conflict"]),
        "struct_consistency_min": float(min(a_struct["format_consistency"], b_struct["format_consistency"])),
        "struct_consistency_prod": float(a_struct["format_consistency"] * b_struct["format_consistency"]),
    }


def context_pair_features(a_meta: Dict[str, object], b_meta: Dict[str, object],
                          block_size: int = 512) -> Dict[str, float]:
    off_a = float(a_meta.get("physical_offset") or 0)
    off_b = float(b_meta.get("physical_offset") or 0)
    dist = abs(off_b - off_a) / block_size
    alloc_a = str(a_meta.get("allocation_status", "raw"))
    alloc_b = str(b_meta.get("allocation_status", "raw"))
    return {
        "physical_dist_log": float(np.log1p(dist)),
        "physical_adjacent": 1.0 if dist == 1.0 else 0.0,
        "physical_close": 1.0 if dist <= 4 else 0.0,
        "cluster_dist": float(np.log1p(abs(float(b_meta.get("cluster") or 0) - float(a_meta.get("cluster") or 0)))),
        "same_cluster": 1.0 if a_meta.get("cluster") == b_meta.get("cluster") else 0.0,
        "alloc_status_match": 1.0 if alloc_a == alloc_b else 0.0,
        "alloc_both_allocated": 1.0 if alloc_a == "allocated" and alloc_b == "allocated" else 0.0,
        "context_compat": float(0.6 * (1.0 - min(1.0, np.log1p(dist) / 8))
                                + 0.4 * (1.0 if alloc_a == alloc_b else 0.0)),
    }


def hist_distances(ha: np.ndarray, hb: np.ndarray) -> Dict[str, float]:
    diff = ha - hb
    l1 = float(0.5 * np.abs(diff).sum())
    cos = _cosine(ha, hb)
    chi2 = float(np.sum(diff ** 2 / (ha + hb + 1e-6)) / len(ha))
    return {"hist_l1": l1, "hist_cosine_sim": cos, "hist_chi2": min(1.0, chi2)}
