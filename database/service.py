"""Database service layer: persist/restore full analysis bundles."""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Analysis, CandidateRow, ExperimentRow, FragmentRow, init_db


def new_analysis_id() -> str:
    return uuid.uuid4().hex[:12]


def create_analysis(session: Session, name: str, image_filename: str,
                    image_size: int, params: Optional[dict] = None) -> str:
    a = Analysis(id=new_analysis_id(), name=name, image_filename=image_filename,
                 image_size_bytes=image_size, status="queued", params=params or {})
    session.add(a)
    session.commit()
    return a.id


def set_status(session: Session, analysis_id: str, status: str,
               error: Optional[str] = None) -> None:
    a = session.get(Analysis, analysis_id)
    if a is None:
        return
    a.status = status
    a.error = error
    session.commit()


def persist_result(session: Session, analysis_id: str, result: Dict[str, object]) -> None:
    """Store a complete :func:`pipeline.analyse_image` result bundle."""
    a = session.get(Analysis, analysis_id)
    if a is None:
        return
    a.status = "done"
    a.finished_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    a.seconds = result.get("seconds", 0.0)
    a.reconstruction_mode = result.get("reconstruction_mode", "")
    a.summary = {
        "n_fragments": len(result["fragments"]),
        "n_candidates": len(result["candidates"]),
        "n_anomalies": result["anomaly"]["flagged"],
        "n_clusters": result["clusters"]["k"],
        "cluster_silhouette": result["clusters"]["silhouette"],
        "graph": {"nodes": len(result["graph"]["nodes"]),
                  "edges": len(result["graph"]["edges"])},
        "image_summary": result["image_summary"],
        "class_counts": _class_counts(result["candidates"]),
    }
    a.metrics = result.get("metrics", {})
    a.graph_data = result["graph"]
    a.cluster_data = result["clusters"]
    a.anomaly_data = result["anomaly"]

    session.query(FragmentRow).filter_by(analysis_id=analysis_id).delete()
    session.query(CandidateRow).filter_by(analysis_id=analysis_id).delete()
    for f in result["fragments"]:
        f_data = {k: v for k, v in f.items() if k != "embedding_preview"}
        session.add(FragmentRow(analysis_id=analysis_id, **f_data))
    for rank, c in enumerate(result["candidates"], 1):
        session.add(CandidateRow(
            analysis_id=analysis_id, rank=rank, candidate_id=c.candidate_id,
            file_type=c.file_type, estimated_size=c.estimated_size,
            fragment_sequence=c.fragment_sequence, gaps=c.gaps,
            confidence=c.confidence, raw_score=c.raw_score,
            classification=c.classification, ml_evidence=c.ml_evidence,
            forensic_evidence=c.forensic_evidence,
            anomaly_analysis=c.anomaly_analysis, issues=c.issues))
    session.commit()


def _class_counts(candidates: List) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in candidates:
        counts[c.classification] = counts.get(c.classification, 0) + 1
    return counts


def list_analyses(session: Session) -> List[Dict[str, object]]:
    rows = session.query(Analysis).order_by(Analysis.created_at.desc()).all()
    return [_analysis_dict(a) for a in rows]


def _analysis_dict(a: Analysis) -> Dict[str, object]:
    return {
        "id": a.id, "name": a.name, "status": a.status, "error": a.error,
        "image_filename": a.image_filename, "image_size_bytes": a.image_size_bytes,
        "reconstruction_mode": a.reconstruction_mode,
        "summary": a.summary or {}, "metrics": a.metrics or {},
        "seconds": a.seconds, "created_at": str(a.created_at),
    }


def get_analysis(session: Session, analysis_id: str) -> Optional[Dict[str, object]]:
    a = session.get(Analysis, analysis_id)
    return _analysis_dict(a) if a else None


def get_fragments(session: Session, analysis_id: str, offset: int = 0,
                  limit: int = 200, anomaly_only: bool = False,
                  fs_cluster: Optional[int] = None) -> List[Dict[str, object]]:
    q = session.query(FragmentRow).filter_by(analysis_id=analysis_id)
    if anomaly_only:
        q = q.filter(FragmentRow.anomaly_status == "SUSPICIOUS")
    if fs_cluster is not None:
        q = q.filter(FragmentRow.fs_cluster_label == fs_cluster)
    q = q.order_by(FragmentRow.index).offset(offset).limit(min(limit, 2000))
    return [{c.name: getattr(r, c.name) for c in FragmentRow.__table__.columns
             if c.name != "embedding"} | {"embedding": r.embedding}
            for r in q.all()]


def get_candidates(session: Session, analysis_id: str) -> List[Dict[str, object]]:
    rows = session.query(CandidateRow).filter_by(analysis_id=analysis_id)\
        .order_by(CandidateRow.rank).all()
    return [{c.name: getattr(r, c.name) for c in CandidateRow.__table__.columns}
            for r in rows]


def save_experiment(session: Session, name: str, data: dict,
                    analysis_id: Optional[str] = None) -> None:
    session.add(ExperimentRow(analysis_id=analysis_id, name=name, data=data))
    session.commit()


def get_experiments(session: Session, analysis_id: Optional[str] = None) -> List[Dict[str, object]]:
    q = session.query(ExperimentRow)
    if analysis_id:
        q = q.filter(ExperimentRow.analysis_id == analysis_id)
    rows = q.order_by(ExperimentRow.created_at.desc()).limit(50).all()
    return [{"id": r.id, "analysis_id": r.analysis_id, "name": r.name,
             "data": r.data, "created_at": str(r.created_at)} for r in rows]


def delete_analysis(session: Session, analysis_id: str) -> bool:
    a = session.get(Analysis, analysis_id)
    if a is None:
        return False
    session.query(FragmentRow).filter_by(analysis_id=analysis_id).delete()
    session.query(CandidateRow).filter_by(analysis_id=analysis_id).delete()
    session.query(ExperimentRow).filter_by(analysis_id=analysis_id).delete()
    session.delete(a)
    session.commit()
    return True
