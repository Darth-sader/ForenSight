"""ForenSight v2 feature engineering package.

Enhancements over v1:
1. Leakage-free feature calculation with partition-isolated normalizer integration.
2. Robust numeric sanitation: all outputs guaranteed free of NaNs, Infs, or invalid bounds.
3. Explicit support for metadata-independent mode where physical offsets are not leaked.
4. Full backward-compatible shape contracts (single: 400 features, pair: 362 features).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

from features.byte_features import BYTE_HIST_DIM, compute_byte_matrix
from features.contextual_features import compute_contextual_matrix, single_contextual_names
from features.local_entropy import (PROFILE_DIM, local_entropy_features,
                                    local_entropy_matrix, local_entropy_pair_features)
from features.pair_features import (_cosine, boundary_byte_compatibility,
                                    context_pair_features, hist_distances,
                                    structural_pair_features)
from features import PAIR_NAMES
from features.statistical_features import compute_statistical_matrix, NGRAM_DIM
from features.structural_features import (SINGLE_STRUCTURAL_NAMES, FORMATS,
                                          compute_structural_matrix, detect_format_markers)
if TYPE_CHECKING:
    from preprocessing_v2.normalize_v2 import FeatureNormalizerV2

FAMILIES = ["byte", "statistical", "structural", "contextual"]
ABLATION_MASKS = {
    "A_byte": ["byte"],
    "B_byte_statistical": ["byte", "statistical"],
    "C_byte_stat_structural": ["byte", "statistical", "structural"],
    "D_plus_boundary": ["byte", "statistical", "structural", "boundary"],
    "E_all_context": ["byte", "statistical", "structural", "boundary", "context"],
}


class FeatureEngineV2:
    """Computes single-fragment and pairwise representations with strict numeric hygiene."""

    def __init__(
        self,
        data_matrix: np.ndarray,
        metas: List[Dict[str, object]],
        normalizer: Optional[FeatureNormalizerV2] = None,
        sanitize_nans: bool = True,
    ):
        if data_matrix.ndim != 2:
            raise ValueError(f"data_matrix must be (n, block_size), got shape {data_matrix.shape}")
        self.data_ = np.asarray(data_matrix, dtype=np.uint8)
        self.metas_ = metas
        self.n_ = data_matrix.shape[0]
        self.normalizer = normalizer
        self.sanitize_nans = sanitize_nans

        byte = compute_byte_matrix(self.data_)
        stat = compute_statistical_matrix(self.data_)
        struct = compute_structural_matrix(self.data_)
        ctx = compute_contextual_matrix(metas)
        le = local_entropy_matrix(self.data_)
        stat.update(le)

        self.cache_ = {"byte": byte, "statistical": stat, "structural": struct, "contextual": ctx}

        names: List[str] = []
        families: List[str] = []
        cols: List[np.ndarray] = []
        for fam in FAMILIES:
            block = self.cache_[fam]
            for key, arr in block.items():
                if arr.ndim == 1:
                    names.append(f"{fam[0]}_{key}")
                    families.append(fam)
                    cols.append(arr)
                else:
                    for d in range(arr.shape[1]):
                        names.append(f"{fam[0]}_{key}_{d}")
                        families.append(fam)
                        cols.append(arr[:, d])

        raw_single = np.column_stack(cols).astype(np.float32)
        if self.sanitize_nans:
            raw_single = np.nan_to_num(raw_single, nan=0.0, posinf=1e6, neginf=-1e6)
        self.single_matrix_ = raw_single
        self.single_names_ = names
        self.single_families_ = families
        self._struct_dicts: List[Optional[Dict[str, float]]] = [None] * self.n_

    # ------------------------------------------------------------- accessors
    def struct_dict(self, i: int) -> Dict[str, float]:
        if self._struct_dicts[i] is None:
            self._struct_dicts[i] = detect_format_markers(self.data_[i])
        return self._struct_dicts[i]

    def single_index(self, name: str) -> int:
        return self.single_names_.index(name)

    # ------------------------------------------------------------- pair features
    def pair_matrix(self, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
        rows = np.zeros((len(pairs), len(PAIR_NAMES)), dtype=np.float32)
        for r, (i, j) in enumerate(pairs):
            rows[r] = self.pair_row(i, j)
        if self.sanitize_nans:
            rows = np.nan_to_num(rows, nan=0.0, posinf=1.0, neginf=-1.0)
        return rows

    def pair_matrix_vec(self, pairs: np.ndarray) -> np.ndarray:
        pairs = np.asarray(pairs, dtype=np.int64)
        if len(pairs) == 0:
            return np.zeros((0, len(PAIR_NAMES)), dtype=np.float32)

        ia, ib = pairs[:, 0], pairs[:, 1]
        byte, stat, struct, ctx = (self.cache_[k] for k in FAMILIES)
        n = len(pairs)
        name_to_pos = {name: k for k, (name, _) in enumerate(PAIR_NAMES)}
        out = np.zeros((n, len(PAIR_NAMES)), dtype=np.float32)

        def put(names_vals: Dict[str, np.ndarray]) -> None:
            for nm, vals in names_vals.items():
                out[:, name_to_pos[nm]] = vals

        # ---- byte family
        ha, hb = byte["hist"][ia], byte["hist"][ib]
        hist_absdiff = np.abs(ha - hb)
        diff = ha - hb
        l1 = 0.5 * np.abs(diff).sum(axis=1)
        den = (np.linalg.norm(ha, axis=1) * np.linalg.norm(hb, axis=1)) + 1e-12
        cos = (ha * hb).sum(axis=1) / den
        chi2 = ((diff ** 2) / (ha + hb + 1e-6)).sum(axis=1) / ha.shape[1]
        put({**{f"hist_absdiff_{d}": hist_absdiff[:, d] for d in range(BYTE_HIST_DIM)},
             "hist_l1": l1, "hist_cosine_sim": cos, "hist_chi2": np.minimum(1.0, chi2)})
        put({f"d_{k}": byte[k][ia] - byte[k][ib]
             for k in ("mean", "variance", "min", "max", "entropy")})
        put({f"p_{k}": byte[k][ia] * byte[k][ib]
             for k in ("mean", "variance", "entropy")})

        # ---- statistical family
        put({"entropy_diff": byte["entropy"][ia] - byte["entropy"][ib]})
        put({f"d_{k}": stat[k][ia] - stat[k][ib]
             for k in ("printable_ratio", "ascii_ratio", "zero_ratio", "high_byte_ratio",
                       "byte_diversity", "top_byte_freq", "repeat_run_ratio", "bigram_entropy",
                       "le_mean", "le_std", "le_min", "le_max", "le_range", "le_gradient",
                       "le_peak_ratio", "le_max_pos", "le_head", "le_tail",
                       "le_low_entropy_fraction")})
        put({"p_printable": stat["printable_ratio"][ia] * stat["printable_ratio"][ib]})
        for ng in ("ngram2", "ngram3"):
            va, vb = stat[ng][ia], stat[ng][ib]
            dn = np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-12
            put({f"{ng}_cosine": (va * vb).sum(axis=1) / dn})
        pa_p, pb_p = stat["le_profile"][ia], stat["le_profile"][ib]
        dn = np.linalg.norm(pa_p, axis=1) * np.linalg.norm(pb_p, axis=1) + 1e-12
        le_cos = (pa_p * pb_p).sum(axis=1) / dn
        le_bdiff = np.abs(stat["le_tail"][ia] - stat["le_head"][ib])
        put({"le_profile_cosine": le_cos,
             "le_boundary_diff": le_bdiff,
             "le_boundary_continuity": np.clip(1.0 - le_bdiff / 4.0, 0.0, 1.0),
             "le_similar_structure": (np.abs(stat["le_mean"][ia] - stat["le_mean"][ib]) < 0.75).astype(float)})

        # ---- structural family
        put({f"d_{k}": struct[k][ia] - struct[k][ib] for k in SINGLE_STRUCTURAL_NAMES})
        put({f"p_{k}": struct[k][ia] * struct[k][ib]
             for k in ("header_detected", "footer_detected", "marker_density",
                       "is_header_fragment", "is_footer_fragment", "format_consistency",
                       "multi_format_conflict")})
        put(self._structural_pair_vec(ia, ib, name_to_pos))

        # ---- boundary family
        put(self._boundary_vec(ia, ib, name_to_pos))

        # ---- contextual family
        put({f"d_{k}": ctx[k][ia] - ctx[k][ib]
             for k in ("alloc_allocated", "alloc_unallocated", "alloc_system",
                       "offset_norm", "cluster_norm", "offset_mod_1000")})
        put({"p_alloc_allocated": ctx["alloc_allocated"][ia] * ctx["alloc_allocated"][ib],
             "p_alloc_system": ctx["alloc_system"][ia] * ctx["alloc_system"][ib]})
        put(self._context_pair_vec(ia, ib, name_to_pos))

        if self.sanitize_nans:
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out.astype(np.float32)

    def _structural_pair_vec(self, ia: np.ndarray, ib: np.ndarray,
                             name_to_pos: Dict[str, int]) -> Dict[str, np.ndarray]:
        struct = self.cache_["structural"]
        types = ("JPEG", "PNG", "PDF", "ZIP", "DOCX", "XLSX", "MP3", "MP4")
        claimed_a = np.zeros((len(ia), len(types)))
        claimed_b = np.zeros((len(ib), len(types)))
        for t_i, t in enumerate(types):
            key = f"header_{t}"
            claimed_a[:, t_i] = (struct[key][ia] >= 0.8) | (struct[f"internal_{t}"][ia] > 0)
            claimed_b[:, t_i] = (struct[key][ib] >= 0.8) | (struct[f"internal_{t}"][ib] > 0)
        both = (claimed_a > 0) & (claimed_b > 0)
        n_a = claimed_a.sum(axis=1)
        same_format = both.sum(axis=1) / np.maximum(1.0, n_a)
        office = np.array([t in ("ZIP", "DOCX", "XLSX") for t in types])
        nonoffice_shared = (both[:, ~office]).any(axis=1)
        office_shared = (claimed_a[:, office].any(axis=1) &
                         claimed_b[:, office].any(axis=1))
        same_family = (nonoffice_shared | office_shared).astype(float)
        return {
            "struct_same_format_detected": np.clip(same_format, 0, 1),
            "struct_same_format_family": same_family,
            "struct_header_then_body": ((struct["is_header_fragment"][ia] >= 1.0) &
                                        (struct["is_header_fragment"][ib] < 1.0)).astype(float),
            "struct_footer_then_any": (struct["is_footer_fragment"][ia] >= 1.0).astype(float),
            "struct_header_in_b": (struct["is_header_fragment"][ib] >= 1.0).astype(float),
            "struct_conflict_sum": (struct["multi_format_conflict"][ia] +
                                    struct["multi_format_conflict"][ib]),
            "struct_consistency_min": np.minimum(struct["format_consistency"][ia],
                                                 struct["format_consistency"][ib]),
            "struct_consistency_prod": struct["format_consistency"][ia] * struct["format_consistency"][ib],
        }

    def _boundary_vec(self, ia: np.ndarray, ib: np.ndarray,
                      name_to_pos: Dict[str, int]) -> Dict[str, np.ndarray]:
        da, db = self.data_[ia], self.data_[ib]
        la, fb = da[:, -1].astype(np.int64), db[:, 0].astype(np.int64)
        struct = self.cache_["structural"]
        pa = (da == la[:, None]).mean(axis=1)
        pb = (db == fb[:, None]).mean(axis=1)
        continuity = 1.0 - np.minimum(1.0, np.abs(pa - pb))
        k = 8
        overlap = (da[:, -k:] == db[:, :k]).mean(axis=1)

        jpegish_a = ((struct["header_JPEG"][ia] > 0) | (struct["internal_JPEG"][ia] > 0) |
                     (la == 0xFF))
        marker_set = np.array([0x01, 0xC0, 0xC1, 0xC2, 0xC4, 0xD0, 0xD1, 0xD2, 0xD3,
                               0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDD,
                               0xE0, 0xE1, 0xFE])
        fb_is_marker = np.isin(fb, marker_set)
        rule = np.where(jpegish_a & (la == 0xFF) & fb_is_marker, 1.0,
                        np.where(jpegish_a & (la != 0xFF) & (fb != 0xFF), 0.8,
                                 np.where(jpegish_a, 0.1, 0.5)))
        endobj = np.frombuffer(b"endobj", dtype=np.uint8)
        pdf_a = struct["internal_PDF"][ia] > 0
        tail6_all = (da[:, -6:] == endobj[None, :]).all(axis=1)
        digit_fb = np.isin(fb, np.frombuffer(b"0123456789<%", dtype=np.uint8).astype(np.int64))
        rule = np.where(pdf_a & tail6_all & digit_fb, np.maximum(rule, 1.0),
                        np.where(pdf_a & tail6_all, np.maximum(rule, 0.3), rule))
        mp3_a = (struct["header_MP3"][ia] > 0) | (struct["internal_MP3"][ia] > 0)
        rule = np.where(mp3_a & (la == 0xFF) & np.isin(fb, [0xFB, 0xFA, 0xF3, 0xF2]),
                        np.maximum(rule, 1.0), rule)
        footer_a = ((struct["footer_PNG"][ia] >= 1.0) | (struct["footer_PDF"][ia] >= 1.0) |
                    (struct["footer_ZIP"][ia] >= 1.0))
        rule = np.where(footer_a & (struct["is_header_fragment"][ib] >= 1.0), 0.0,
                        np.where(footer_a, np.minimum(rule, 0.2), rule))
        header_b = struct["is_header_fragment"][ib] >= 1.0
        header_a = struct["is_header_fragment"][ia] >= 1.0
        rule = np.where(header_b & ~header_a, np.minimum(rule, 0.15), rule)
        generic = np.clip(0.5 * rule + 0.3 * continuity + 0.2, 0.0, 1.0)
        return {
            "boundary_byte_continuity": continuity,
            "prefix_suffix_overlap": overlap,
            "boundary_format_rule": rule,
            "boundary_incomplete_marker": ((la == 0xFF) |
                                           (da[:, -4:] == np.array([0, 0, 0, 1],
                                                                   dtype=np.uint8)).all(axis=1)).astype(float),
            "boundary_frame_alignment": (np.isin(la, [0xFB, 0xFA, 0xF3, 0xF2]) &
                                         (fb == 0xFF)).astype(float),
            "boundary_generic_score": generic,
        }

    def _context_pair_vec(self, ia: np.ndarray, ib: np.ndarray,
                          name_to_pos: Dict[str, int]) -> Dict[str, np.ndarray]:
        off_a = np.array([float(self.metas_[i].get("physical_offset") or 0) for i in ia])
        off_b = np.array([float(self.metas_[i].get("physical_offset") or 0) for i in ib])
        cl_a = np.array([float(self.metas_[i].get("cluster") or 0) for i in ia])
        cl_b = np.array([float(self.metas_[i].get("cluster") or 0) for i in ib])
        alloc_a = np.array([str(self.metas_[i].get("allocation_status", "unallocated")) for i in ia])
        alloc_b = np.array([str(self.metas_[i].get("allocation_status", "unallocated")) for i in ib])
        block = max(1.0, float(self.data_.shape[1]))
        dist = np.abs(off_b - off_a) / block

        return {
            "physical_dist_log": np.log1p(dist),
            "physical_adjacent": (dist == 1.0).astype(float),
            "physical_close": (dist <= 4.0).astype(float),
            "cluster_dist": np.log1p(np.abs(cl_b - cl_a)),
            "same_cluster": (cl_a == cl_b).astype(float),
            "alloc_status_match": (alloc_a == alloc_b).astype(float),
            "alloc_both_allocated": ((alloc_a == "allocated") & (alloc_b == "allocated")).astype(float),
            "context_compat": (0.6 * (1.0 - np.minimum(1.0, np.log1p(dist) / 8)) +
                               0.4 * (alloc_a == alloc_b)),
        }

    def pair_row(self, i: int, j: int) -> np.ndarray:
        a_bytes, b_bytes = self.data_[i], self.data_[j]
        a_s, b_s = self.struct_dict(i), self.struct_dict(j)
        byte, stat, struct, ctx = (self.cache_[k] for k in FAMILIES)
        feats: Dict[str, float] = {}

        # ---- byte family
        hist_absdiff = np.abs(byte["hist"][i] - byte["hist"][j])
        for d in range(BYTE_HIST_DIM):
            feats[f"hist_absdiff_{d}"] = float(hist_absdiff[d])
        feats.update(hist_distances(byte["hist"][i], byte["hist"][j]))
        for k in ("mean", "variance", "min", "max", "entropy"):
            feats[f"d_{k}"] = float(byte[k][i] - byte[k][j])
        for k in ("mean", "variance", "entropy"):
            feats[f"p_{k}"] = float(byte[k][i] * byte[k][j])

        # ---- statistical family
        feats["entropy_diff"] = float(byte["entropy"][i] - byte["entropy"][j])
        for k in ("printable_ratio", "ascii_ratio", "zero_ratio", "high_byte_ratio",
                  "byte_diversity", "top_byte_freq", "repeat_run_ratio", "bigram_entropy",
                  "le_mean", "le_std", "le_min", "le_max", "le_range", "le_gradient",
                  "le_peak_ratio", "le_max_pos", "le_head", "le_tail",
                  "le_low_entropy_fraction"):
            feats[f"d_{k}"] = float(stat[k][i] - stat[k][j])
        feats["p_printable"] = float(stat["printable_ratio"][i] * stat["printable_ratio"][j])
        feats["ngram2_cosine"] = _cosine(stat["ngram2"][i], stat["ngram2"][j])
        feats["ngram3_cosine"] = _cosine(stat["ngram3"][i], stat["ngram3"][j])
        feats.update(local_entropy_pair_features(stat["le_profile"][i], stat["le_profile"][j],
                                                 float(stat["le_tail"][i]), float(stat["le_head"][j])))

        # ---- structural family
        s_keys = SINGLE_STRUCTURAL_NAMES
        for k in s_keys:
            feats[f"d_{k}"] = float(struct[k][i] - struct[k][j])
        for k in ("header_detected", "footer_detected", "marker_density",
                  "is_header_fragment", "is_footer_fragment", "format_consistency",
                  "multi_format_conflict"):
            feats[f"p_{k}"] = float(struct[k][i] * struct[k][j])
        feats.update(structural_pair_features(a_s, b_s))

        # ---- boundary family
        feats.update(boundary_byte_compatibility(a_bytes, b_bytes, a_s, b_s))

        # ---- contextual family
        for k in ("alloc_allocated", "alloc_unallocated", "alloc_system", "offset_norm",
                  "cluster_norm", "offset_mod_1000"):
            feats[f"d_{k}"] = float(ctx[k][i] - ctx[k][j])
        for k in ("alloc_allocated", "alloc_system"):
            feats[f"p_{k}"] = float(ctx[k][i] * ctx[k][j])
        feats.update(context_pair_features(self.metas_[i], self.metas_[j], self.data_.shape[1]))

        row = np.array([feats[name] for name, _ in PAIR_NAMES], dtype=np.float32)
        if self.sanitize_nans:
            row = np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=-1.0)
        return row

    # ------------------------------------------------------------- normalizer helpers
    def fit_normalizer(self, X_train: np.ndarray, method: str = "standard") -> "FeatureNormalizerV2":
        from preprocessing_v2.normalize_v2 import FeatureNormalizerV2
        self.normalizer = FeatureNormalizerV2(method=method)
        self.normalizer.fit(X_train)
        return self.normalizer

    def transform_pairs(self, X: np.ndarray) -> np.ndarray:
        if self.normalizer is None or not self.normalizer.fitted_:
            raise RuntimeError("Cannot transform pairs without a fitted normalizer in FeatureEngineV2.")
        return self.normalizer.transform(X)


__all__ = ["FeatureEngineV2", "PAIR_NAMES", "ABLATION_MASKS", "FAMILIES"]
