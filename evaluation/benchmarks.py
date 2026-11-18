"""Conventional signature-based carving baseline + incremental component
benchmarks (doc §34): traditional carving vs ML relationship model vs ML+graph
vs ML+graph+constraints vs full ForenSight."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from features.structural_features import FORMATS, MARKERS, OFFICE_TYPES


class SignatureCarver:
    """Classic header->footer contiguous carving (doc §34 baseline)."""

    def __init__(self, block_size: int = 512):
        self.block_size = block_size

    def carve(self, data_matrix: np.ndarray, metas: List[dict]) -> List[Dict[str, object]]:
        """Scan blocks for headers, then read forward until footer or gap limit."""
        n = len(data_matrix)
        results: List[Dict[str, object]] = []
        used = np.zeros(n, dtype=bool)
        for i in range(n):
            buf = data_matrix[i].tobytes()
            claimed = None
            for t in FORMATS:
                for marker in MARKERS[t]["header"]:
                    if buf[:len(marker)] == marker:
                        claimed = t
                        break
                if claimed:
                    break
            if not claimed or used[i]:
                continue
            # walk forward contiguously for footer evidence
            seq = [i]
            used[i] = True
            footer_found = False
            for j in range(i + 1, min(n, i + 4000)):
                if used[j]:
                    break
                jbuf = data_matrix[j].tobytes()
                footer = any(jbuf.rfind(m) >= len(jbuf) - 16
                             for m in MARKERS[claimed]["footer"])
                seq.append(j)
                used[j] = True
                if footer:
                    footer_found = True
                    break
            if footer_found or len(seq) == 1:
                results.append({
                    "file_type": claimed, "sequence": seq,
                    "footer_found": footer_found,
                    "method": "signature_carving",
                })
        return results


def carve_candidates_to_sequences(carved: List[Dict[str, object]],
                                  id_of) -> List[List[str]]:
    return [[id_of(i) for i in c["sequence"]] for c in carved]


# --------------------------------------------------------------------------
# Incremental component comparison (doc §34)
# --------------------------------------------------------------------------

def benchmark_components(engine_data, true_sequences, id_of,
                         ml_sequences: Optional[List[List[str]]] = None,
                         graph_sequences: Optional[List[List[str]]] = None,
                         constrained_sequences: Optional[List[List[str]]] = None,
                         full_sequences: Optional[List[List[str]]] = None) -> Dict[str, object]:
    """Recovery rates for each pipeline stage.

    Stage fallbacks: when a stage's sequences are not supplied, the previous
    stage's result is reused so the comparison is strictly incremental.
    """
    from evaluation.metrics import reconstruction_metrics

    stages = {
        "traditional_carving": None,
        "ml_relationship": ml_sequences,
        "ml_graph": graph_sequences,
        "ml_graph_constraints": constrained_sequences,
        "full_forensight": full_sequences,
    }
    out: Dict[str, object] = {}
    prev = None
    for stage, seqs in stages.items():
        if seqs is None:
            seqs = prev or []
        prev = seqs
        out[stage] = reconstruction_metrics(true_sequences, seqs)
    return out
