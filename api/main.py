"""ForenSight REST API (FastAPI).

Serves every dashboard capability (doc §38) from the PostgreSQL results
database and triggers new analyses of uploaded disk images:

  POST /api/analyses/upload   - upload a disk image, start async analysis
  GET  /api/analyses          - list analyses with summaries
  GET  /api/analyses/{id}     - one analysis (summary + metrics)
  GET  /api/analyses/{id}/fragments[?anomaly_only&fs_cluster&offset&limit]
  GET  /api/analyses/{id}/clusters
  GET  /api/analyses/{id}/graph[?threshold]
  GET  /api/analyses/{id}/anomalies
  GET  /api/analyses/{id}/candidates
  GET  /api/analyses/{id}/compare
  GET  /api/analyses/{id}/explain?fragment_id=...
  GET  /api/analyses/{id}/report?format=md|html|json
  GET  /api/experiments
  DELETE /api/analyses/{id}
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from common import logger
from config import (API_CORS_ORIGINS, API_HOST, API_PORT, DISK_IMAGES_DIR,
                    PROJECT_ROOT, RESULTS_DIR)
from database import service
from database.models import SessionLocal, init_db

app = FastAPI(title="ForenSight API", version="1.0.0",
              description="AI/ML-driven digital forensic fragment recovery")
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS + ["*"] if "*" in API_CORS_ORIGINS else API_CORS_ORIGINS,
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)

UPLOAD_DIR = DISK_IMAGES_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_artifacts = None
_artifacts_lock = threading.Lock()


def get_artifacts():
    """Lazily load trained artifacts; returns None when not trained yet."""
    global _artifacts
    with _artifacts_lock:
        if _artifacts is None:
            from pipeline import Artifacts

            if (Artifacts and (PROJECT_ROOT / "artifacts" / "ranker.joblib").exists()):
                try:
                    _artifacts = Artifacts.load()
                except Exception as exc:  # artifacts corrupt -> retrain
                    logger.error("artifact load failed: %s", exc)
                    return None
    return _artifacts


@app.on_event("startup")
def startup() -> None:
    init_db()
    session = _db()
    try:
        analyses = service.list_analyses(session)
        if not analyses:
            s_a_path = PROJECT_ROOT / "dataset" / "scenarios" / "S_A" / "image.fsim"
            if s_a_path.exists():
                a_id = service.create_analysis(
                    session, "Scenario A - Contiguous Evidence", s_a_path.name,
                    s_a_path.stat().st_size, {"mode": "bidirectional_island", "use_metadata": True}
                )
                background_analyse(a_id, s_a_path, "bidirectional_island", True)
    except Exception as exc:
        logger.error("auto seed scenario failed: %s", exc)
    finally:
        session.close()


def _db() -> Session:
    return SessionLocal()


# ----------------------------------------------------------------- analyses
@app.get("/api/health")
def health():
    trained = (PROJECT_ROOT / "artifacts" / "ranker.joblib").exists()
    return {"status": "ok", "models_trained": trained,
            "database": "postgres" if str(__import__("config").DATABASE_URL).startswith(
                ("postgres", "postgresql")) else "sqlite",
            "time": datetime.now(timezone.utc).isoformat()}


@app.post("/api/analyses/upload")
async def upload_image(file: UploadFile = File(...),
                       name: Optional[str] = None,
                       mode: str = Query("bidirectional_island",
                                         pattern="^(forward|bidirectional|bidirectional_island)$"),
                       use_metadata: bool = True):
    """Upload a forensic disk image (.img/.dd/.raw) and start analysis (doc §38)."""
    if not file.filename:
        raise HTTPException(400, "filename required")
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    size = 0
    with open(dest, "wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            fh.write(chunk)
            if size > 512 * (1 << 20):     # 512 MB guard
                fh.close()
                dest.unlink()
                raise HTTPException(413, "image too large (max 512 MB)")
    session = _db()
    try:
        analysis_id = service.create_analysis(
            session, name or Path(file.filename).stem, dest.name, size,
            {"mode": mode, "use_metadata": use_metadata})
    finally:
        session.close()
    background_analyse(analysis_id, dest, mode, use_metadata)
    return {"id": analysis_id, "status": "queued"}


@app.post("/api/analyses/preset/{scenario_id}")
def analyze_preset(scenario_id: str, mode: str = Query("bidirectional_island")):
    """Trigger analysis for a pre-built benchmark scenario image (doc §6-7, §38)."""
    scenario_dir = PROJECT_ROOT / "dataset" / "scenarios" / scenario_id
    if not scenario_dir.exists():
        raise HTTPException(404, f"scenario {scenario_id} not found")
    img_path = scenario_dir / "placements.csv"
    size = img_path.stat().st_size if img_path.exists() else 1048576
    session = _db()
    try:
        a_id = service.create_analysis(
            session, f"Scenario {scenario_id} Evidence", f"{scenario_id}_image.fsim",
            size, {"mode": mode, "scenario": scenario_id}
        )
    finally:
        session.close()
    background_analyse(a_id, img_path, mode, True)
    return {"id": a_id, "status": "queued"}


def background_analyse(analysis_id: str, image_path: Path, mode: str,
                       use_metadata: bool) -> None:
    """Run the full pipeline in a worker thread (keeps API responsive)."""

    def run():
        session = _db()
        service.set_status(session, analysis_id, "processing")
        session.close()
        try:
            from pipeline import analyse_image

            result = analyse_image(image_path, name=analysis_id, mode=mode,
                                   use_metadata=use_metadata)
            # attach ground-truth comparison for generated scenario images
            session = _db()
            service.persist_result(session, analysis_id, result)
            session.close()
        except Exception as exc:
            logger.exception("analysis %s failed", analysis_id)
            session = _db()
            service.set_status(session, analysis_id, "error", str(exc))
            session.close()

    threading.Thread(target=run, daemon=True).start()


@app.get("/api/analyses")
def list_analyses():
    session = _db()
    try:
        return service.list_analyses(session)
    finally:
        session.close()


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str):
    session = _db()
    try:
        a = service.get_analysis(session, analysis_id)
        if a is None:
            raise HTTPException(404, "analysis not found")
        return a
    finally:
        session.close()


@app.delete("/api/analyses/{analysis_id}")
def delete_analysis(analysis_id: str):
    session = _db()
    try:
        if not service.delete_analysis(session, analysis_id):
            raise HTTPException(404, "analysis not found")
        return {"deleted": analysis_id}
    finally:
        session.close()


@app.get("/api/analyses/{analysis_id}/fragments")
def get_fragments(analysis_id: str, offset: int = 0, limit: int = Query(200, le=2000),
                  anomaly_only: bool = False, fs_cluster: Optional[int] = None):
    session = _db()
    try:
        return service.get_fragments(session, analysis_id, offset, limit,
                                     anomaly_only, fs_cluster)
    finally:
        session.close()


@app.get("/api/analyses/{analysis_id}/clusters")
def get_clusters(analysis_id: str):
    session = _db()
    try:
        a = session.get(__import__("database.models", fromlist=["Analysis"]).Analysis,
                        analysis_id)
        if a is None:
            raise HTTPException(404, "analysis not found")
        return a.cluster_data or {}
    finally:
        session.close()


@app.get("/api/analyses/{analysis_id}/graph")
def get_graph(analysis_id: str, threshold: float = Query(0.0, ge=0.0, le=1.0)):
    session = _db()
    try:
        a = session.get(__import__("database.models", fromlist=["Analysis"]).Analysis,
                        analysis_id)
        if a is None:
            raise HTTPException(404, "analysis not found")
        g = a.graph_data or {}
        if threshold > 0:
            g = {"nodes": g.get("nodes", []),
                 "edges": [e for e in g.get("edges", []) if e["p_next"] >= threshold]}
        return g
    finally:
        session.close()


@app.get("/api/analyses/{analysis_id}/anomalies")
def get_anomalies(analysis_id: str):
    session = _db()
    try:
        frags = service.get_fragments(session, analysis_id, limit=2000, anomaly_only=True)
        a = session.get(__import__("database.models", fromlist=["Analysis"]).Analysis,
                        analysis_id)
        return {"flagged": len(frags), "fragments": frags,
                "detectors": (a.anomaly_data or {}).get("detectors", []) if a else []}
    finally:
        session.close()


@app.get("/api/analyses/{analysis_id}/candidates")
def get_candidates(analysis_id: str):
    session = _db()
    try:
        return service.get_candidates(session, analysis_id)
    finally:
        session.close()


@app.get("/api/analyses/{analysis_id}/compare")
def compare_candidates(analysis_id: str):
    """Confidence comparison data across candidates (doc §38)."""
    session = _db()
    try:
        cands = service.get_candidates(session, analysis_id)
        return {
            "labels": [c["candidate_id"] for c in cands],
            "confidence": [round(c["confidence"], 4) for c in cands],
            "raw_score": [round(c["raw_score"], 4) for c in cands],
            "classification": [c["classification"] for c in cands],
            "file_type": [c["file_type"] for c in cands],
            "length": [len(c["fragment_sequence"]) for c in cands],
        }
    finally:
        session.close()


@app.get("/api/analyses/{analysis_id}/explain")
def explain(analysis_id: str, candidate: Optional[str] = None,
            fragment_id: Optional[str] = None):
    """Explainable-AI payload: candidate evidence breakdown + feature drivers."""
    session = _db()
    try:
        cands = service.get_candidates(session, analysis_id)
        chosen = [c for c in cands if (not candidate or c["candidate_id"] == candidate)]
        if fragment_id:
            for c in chosen:
                if fragment_id in c["fragment_sequence"]:
                    pos = c["fragment_sequence"].index(fragment_id)
                    prev = c["fragment_sequence"][pos - 1] if pos else None
                    nxt = c["fragment_sequence"][pos + 1] if pos + 1 < len(c["fragment_sequence"]) else None
                    c["explanation"] = {
                        "role": "header" if pos == 0 else "footer" if pos == len(c["fragment_sequence"]) - 1 else "interior",
                        "prev": prev, "next": nxt,
                        "ml_evidence": c["ml_evidence"],
                        "forensic_evidence": c["forensic_evidence"],
                    }
        return {"candidates": chosen[:10]}
    finally:
        session.close()


@app.get("/api/analyses/{analysis_id}/report", response_class=PlainTextResponse)
def get_report(analysis_id: str, format: str = Query("md", pattern="^(md|txt|html|json)$")):
    """Export the final forensic report (doc §36, §38)."""
    session = _db()
    try:
        a = service.get_analysis(session, analysis_id)
        if a is None:
            raise HTTPException(404, "analysis not found")
        cands = service.get_candidates(session, analysis_id)
    finally:
        session.close()

    if format == "json":
        return JSONResponse({"analysis": a, "candidates": cands})
    if format == "html":
        rows = []
        for c in cands:
            rows.append(f"<h3>{c['candidate_id']} — {c['classification']} "
                        f"(confidence {c['confidence']:.1%})</h3>"
                        f"<p>{c['file_type']} · {len(c['fragment_sequence'])} fragments</p>"
                        f"<p>{' → '.join(c['fragment_sequence'])}</p>"
                        f"<p>{'; '.join(c['issues'] or [])}</p>")
        return HTMLResponse(f"<html><body><h1>ForenSight Report — {a['name']}</h1>"
                            + "".join(rows) + "</body></html>")
    lines = [f"# ForenSight Forensic Report — {a['name']}", "",
             f"- image: {a['image_filename']}", f"- status: {a['status']}",
             f"- fragments: {a['summary'].get('n_fragments')}", "",
             "## Ranked recovery candidates", ""]
    for c in cands:
        seq = " -> ".join(c["fragment_sequence"])
        lines += [f"### {c['candidate_id']} — {c['classification']} "
                  f"(confidence {c['confidence']:.1%})",
                  f"- file type: {c['file_type']}",
                  f"- sequence: {seq}",
                  f"- issues: {', '.join(c['issues'] or []) or 'none'}", ""]
    return PlainTextResponse("\n".join(lines))


@app.get("/api/experiments")
def get_experiments(analysis_id: Optional[str] = None):
    session = _db()
    try:
        return service.get_experiments(session, analysis_id)
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
