"""Structural features (doc §9A): file-format markers, header/footer evidence,
internal structural markers and format-specific consistency checks."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

FORMATS = ["JPEG", "PNG", "PDF", "ZIP", "DOCX", "XLSX", "MP3", "MP4"]

MARKERS: Dict[str, Dict[str, List[bytes]]] = {
    "JPEG": {"header": [b"\xff\xd8\xff"], "footer": [b"\xff\xd9"],
             "internal": [b"\xff\xda", b"\xff\xdb", b"\xff\xc0", b"\xff\xc2",
                          b"\xff\xc4", b"JFIF", b"Exif"]},
    "PNG": {"header": [b"\x89PNG\r\n\x1a\n"], "footer": [b"IEND"],
            "internal": [b"IHDR", b"IDAT", b"PLTE", b"pHYs", b"tEXt"]},
    "PDF": {"header": [b"%PDF-"], "footer": [b"%%EOF"],
            "internal": [b"obj", b"endobj", b"stream", b"xref", b"trailer"]},
    "ZIP": {"header": [b"PK\x03\x04"], "footer": [b"PK\x05\x06"],
            "internal": [b"PK\x01\x02", b"PK\x07\x08"]},
    "DOCX": {"header": [b"PK\x03\x04"], "footer": [b"PK\x05\x06"],
             "internal": [b"word/document.xml", b"[Content_Types].xml", b"w:document"]},
    "XLSX": {"header": [b"PK\x03\x04"], "footer": [b"PK\x05\x06"],
             "internal": [b"xl/workbook.xml", b"[Content_Types].xml", b"worksheet"]},
    "MP3": {"header": [b"ID3", b"\xff\xfb", b"\xff\xfa"], "footer": [],
            "internal": [b"\xff\xfb", b"\xff\xfa", b"\xff\xf3", b"\xff\xf2"]},
    "MP4": {"header": [b"ftyp"], "footer": [],
            "internal": [b"moov", b"mdat", b"trak", b"avc1", b"mdhd"]},
}
OFFICE_TYPES = {"DOCX", "XLSX", "ZIP"}


def detect_format_markers(data: np.ndarray) -> Dict[str, float]:
    """Header/footer/internal marker evidence for one fragment (raw bytes)."""
    buf = data.tobytes()
    n = len(buf)
    feats: Dict[str, float] = {}
    header_hits: List[str] = []
    for t in FORMATS:
        m = MARKERS[t]
        h_score = 0.0
        for marker in m["header"]:
            pos = buf.find(marker)
            if pos == 0:
                h_score = max(h_score, 1.0)
            elif 0 < pos <= 16:
                h_score = max(h_score, 0.8)     # MP4 box size prefix, ID3 padding...
            elif pos > 0:
                h_score = max(h_score, 0.4)
        feats[f"header_{t}"] = h_score
        f_score = 0.0
        for marker in m["footer"]:
            pos = buf.rfind(marker)
            if pos >= 0:
                f_score = max(f_score, 1.0 if pos >= n - 16 else 0.4)
        feats[f"footer_{t}"] = f_score
        internal = sum(buf.count(x) for x in m["internal"])
        feats[f"internal_{t}"] = min(1.0, internal / 6.0)
        if h_score >= 0.8:
            header_hits.append(t)
    feats["header_detected"] = 1.0 if header_hits else 0.0
    feats["footer_detected"] = 1.0 if any(feats[f"footer_{t}"] >= 1.0 for t in FORMATS) else 0.0
    feats["marker_density"] = min(1.0, sum(feats[f"internal_{t}"] for t in FORMATS) / 3.0)
    feats["is_header_fragment"] = 1.0 if header_hits else 0.0
    feats["is_footer_fragment"] = 1.0 if any(feats[f"footer_{t}"] >= 1.0 for t in FORMATS) else 0.0
    feats["format_consistency"] = _format_consistency(buf, data)
    feats["multi_format_conflict"] = _multi_conflict(feats)
    return feats


def _multi_conflict(feats: Dict[str, float]) -> float:
    """More than one distinct header family claiming this fragment."""
    families = set()
    for t in FORMATS:
        if feats[f"header_{t}"] >= 0.8:
            families.add("office" if t in OFFICE_TYPES else t)
    return 1.0 if len(families) > 1 else 0.0


def _format_consistency(buf: bytes, data: np.ndarray) -> float:
    """Format-specific structural sanity of the fragment content (0..1)."""
    score = 0.5
    if buf[:3] == b"\xff\xd8\xff":                      # JPEG start-of-image
        valid = bad = 0
        for i in range(2, min(len(buf) - 1, 512)):
            if buf[i] == 0xFF:
                nxt = buf[i + 1]
                if nxt == 0x00 or nxt == 0xFF or 0xD0 <= nxt <= 0xD9 or nxt in (
                        0x01, 0xC0, 0xC1, 0xC2, 0xC4, 0xDA, 0xDB, 0xDD, 0xE0, 0xE1, 0xFE):
                    valid += 1
                else:
                    bad += 1
        if valid + bad:
            score = max(score, 0.9 if bad == 0 else 0.9 * valid / (valid + bad))
    elif buf[:8] == b"\x89PNG\r\n\x1a\n":               # PNG: IHDR chunk must follow
        score = 0.95 if buf[12:16] == b"IHDR" else 0.7
    elif buf[:5] == b"%PDF-":                           # PDF header + indirect object
        score = 0.95 if b"obj" in buf[:64] else 0.7
    elif buf[:4] == b"PK\x03\x04":                      # ZIP/DOCX/XLSX local header
        ok = b"[Content_Types].xml" in buf or b"word/" in buf or b"xl/" in buf
        score = 0.9 if ok else 0.75
    elif buf[:3] == b"ID3" or (buf[0] == 0xFF and 0xFA >= buf[1] >= 0xF2 and (buf[1] & 0x06)):
        score = 0.9                                     # MP3 tag / frame sync
    elif buf[4:8] == b"ftyp":
        score = 0.95                                    # MP4 ftyp box
    else:
        # generic consistency: JPEG-marker pollution inside non-JPEG content
        ff_pollution = float(np.mean((data[:-1] == 0xFF) & (data[1:] >= 0xC0) & (data[1:] <= 0xFE)))
        score = 0.5 - min(0.3, ff_pollution * 8)
    return float(np.clip(score, 0.0, 1.0))


def compute_structural_matrix(data_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    n = data_matrix.shape[0]
    names = _feature_names()
    out = {k: np.zeros(n) for k in names}
    for i in range(n):
        feats = detect_format_markers(data_matrix[i])
        for k in names:
            out[k][i] = feats.get(k, 0.0)
    return out


def _feature_names() -> List[str]:
    names: List[str] = []
    for t in FORMATS:
        names += [f"header_{t}", f"footer_{t}", f"internal_{t}"]
    names += ["header_detected", "footer_detected", "marker_density",
              "is_header_fragment", "is_footer_fragment", "format_consistency",
              "multi_format_conflict"]
    return names


SINGLE_STRUCTURAL_NAMES = _feature_names()
