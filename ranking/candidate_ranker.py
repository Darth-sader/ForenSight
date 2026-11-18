"""Candidate ranking (doc §21, §36): rank reconstruction candidates with a
learned ranking model (learning-to-rank over per-candidate evidence) and
emit final forensic output records with calibrated confidence bands.

Supports three modes:
* heuristic  - fixed weighted score (doc §20)
* ltr        learned logistic meta-model over evidence features
* full       ltr when trained, heuristic fallback
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import ReconstructionCandidate
from config import CONFIDENCE_BANDS


EVIDENCE_FEATURES = [
    "transition", "embedding", "structural", "context", "anomaly",
    "gap_ratio", "violations", "length", "seed_rank", "min_edge",
]


def candidate_evidence_row(h: dict) -> np.ndarray:
    c = h["components"]
    return np.array([
        c["transition"], c["embedding"], c["structural"], c["context"],
        c.get("anomaly", 0.0), c.get("gap_ratio", 0.0), float(h["violations"]),
        float(np.log1p(len(h["path"].nodes))), 1.0 / (1.0 + h["rank_in_seed"]),
        h["path"].min_score,
    ], dtype=np.float64)


class CandidateRanker:
    """Ranks candidate reconstructions and produces final output (doc §36)."""

    def __init__(self):
        self.meta: Optional[LogisticRegression] = None   # learning-to-rank meta-model

    # ------------------------------------------------------------------
    def fit_ltr(self, hypotheses: List[dict], labels: List[int]) -> "CandidateRanker":
        """Train the meta ranker: label 1 = candidate matches ground truth file."""
        X = np.stack([candidate_evidence_row(h) for h in hypotheses])
        y = np.asarray(labels)
        if len(set(y.tolist())) < 2:
            return self
        self.meta = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.meta.fit(X, y)
        return self

    def _score(self, h: dict) -> float:
        if self.meta is not None:
            row = candidate_evidence_row(h).reshape(1, -1)
            return float(self.meta.predict_proba(row)[0, 1])
        return float(h["raw_score"])

    # ------------------------------------------------------------------
    def rank(self, hypotheses: List[dict], fragment_id_of, file_type_of=None,
             anomaly_of: Optional[Dict[int, float]] = None,
             calibrated_confidence_of: Optional[callable] = None,
             top_n: int = 50) -> List[ReconstructionCandidate]:
        # penalise candidates whose fragments already appear in a better candidate
        used: set = set()
        ranked: List[dict] = sorted(hypotheses, key=lambda h: -self._score(h))
        out: List[ReconstructionCandidate] = []
        for h in ranked:
            nodes = h["path"].nodes
            if len(nodes) < 2:
                continue
            overlap = len(set(nodes) & used)
            reused = len(set(nodes)) != len(nodes)
            score = self._score(h) - 0.15 * overlap - (0.25 if reused else 0.0)
            ftype = (file_type_of(nodes[0]) if file_type_of else "UNKNOWN")
            raw_conf = calibrated_confidence_of(h) if calibrated_confidence_of else score
            if raw_conf is None:
                raw_conf = score if score is not None else 0.5
            confidence = float(np.clip(raw_conf - 0.1 * h["violations"] - 0.08 * overlap, 0.0, 1.0))
            classification = self.classify(confidence)
            ml_evidence = {
                "transition_mean": round(float(h["components"]["transition"]), 4),
                "embedding_similarity": round(float(h["components"]["embedding"]), 4),
                "structural_consistency": round(float(h["components"]["structural"]), 4),
                "contextual_evidence": round(float(h["components"]["context"]), 4),
                "min_edge_score": round(float(h["path"].min_score), 4),
            }
            anomalies = []
            if anomaly_of:
                worst = sorted(nodes, key=lambda n: -anomaly_of.get(n, 0.0))[:3]
                anomalies = [f"{fragment_id_of(n)}: {anomaly_of.get(n, 0.0):.2f}"
                             for n in worst if anomaly_of.get(n, 0.0) > 0.7]
            issues = list(h["issues"])
            if overlap:
                issues.append(f"Competing fragment assignment ({overlap} fragments shared)")
            cand = ReconstructionCandidate(
                candidate_id="", file_type=ftype,
                fragment_sequence=[fragment_id_of(n) for n in nodes],
                estimated_size=512 * len(nodes),
                confidence=confidence, raw_score=score,
                classification=classification,
                ml_evidence=ml_evidence,
                forensic_evidence=dict(h.get("evidence", {})),
                anomaly_analysis={"max_anomaly": round(max([anomaly_of.get(n, 0.0) for n in nodes]), 4)
                                  if anomaly_of else 0.0,
                                  "flagged": anomalies},
                issues=issues,
                gaps=list(h["path"].gaps),
            )
            out.append(cand)
            used.update(nodes)
            if len(out) >= top_n:
                break
        for i, c in enumerate(out, 1):
            c.candidate_id = f"RC-{i:03d}"
        return out

    @staticmethod
    def classify(confidence: float) -> str:
        if confidence >= CONFIDENCE_BANDS["HIGH-CONFIDENCE RECOVERY"]:
            return "HIGH-CONFIDENCE RECOVERY"
        if confidence >= CONFIDENCE_BANDS["UNCERTAIN"]:
            return "UNCERTAIN"
        return "LOW-CONFIDENCE RECOVERY"
