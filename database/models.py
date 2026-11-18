"""SQLAlchemy ORM models - ForenSight results database (PostgreSQL primary,
SQLite fallback for zero-setup local runs)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, BigInteger, Boolean, Column, DateTime, Float,
                        Integer, String, Text, create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

Base = declarative_base()


def _now():
    return datetime.now(timezone.utc)


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String(40), primary_key=True)
    name = Column(String(200), nullable=False)
    image_filename = Column(String(400))
    image_size_bytes = Column(BigInteger)
    status = Column(String(20), default="queued")   # queued|processing|done|error
    error = Column(Text, nullable=True)
    reconstruction_mode = Column(String(40), default="bidirectional_island")
    params = Column(JSON, default=dict)
    summary = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    graph_data = Column(JSON, default=dict)
    cluster_data = Column(JSON, default=dict)
    anomaly_data = Column(JSON, default=dict)
    seconds = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=_now)
    finished_at = Column(DateTime(timezone=True), nullable=True)


class FragmentRow(Base):
    __tablename__ = "fragments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(40), index=True, nullable=False)
    fragment_id = Column(String(20), index=True)
    index = Column(Integer)
    source_file = Column(String(200), nullable=True)
    file_type = Column(String(20), nullable=True)
    original_position = Column(Integer, nullable=True)
    fragment_size = Column(Integer)
    physical_offset = Column(BigInteger)
    cluster = Column(BigInteger)
    allocation_status = Column(String(20))
    fs_cluster_label = Column(Integer)
    anomaly_score = Column(Float)
    anomaly_status = Column(String(20))
    entropy = Column(Float)
    header_evidence = Column(Float)
    footer_evidence = Column(Float)
    format_consistency = Column(Float)
    embedding = Column(JSON)


class CandidateRow(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(40), index=True, nullable=False)
    rank = Column(Integer)
    candidate_id = Column(String(20))
    file_type = Column(String(20))
    estimated_size = Column(BigInteger)
    fragment_sequence = Column(JSON)
    gaps = Column(JSON, default=list)
    confidence = Column(Float)
    raw_score = Column(Float)
    classification = Column(String(40))
    ml_evidence = Column(JSON)
    forensic_evidence = Column(JSON)
    anomaly_analysis = Column(JSON)
    issues = Column(JSON)


class ExperimentRow(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(40), nullable=True, index=True)
    name = Column(String(120), index=True)
    data = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=_now)


def make_engine(url: str = DATABASE_URL):
    """Create the SQLAlchemy engine; PostgreSQL when DATABASE_URL is a
    postgres:// URL, SQLite otherwise (zero-setup fallback)."""
    if url.startswith("postgres://"):  # common cloud-provider shorthand
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    kwargs = {"pool_pre_ping": True} if url.startswith("postgresql") else {}
    return create_engine(url, **kwargs)


ENGINE = make_engine()
SessionLocal = sessionmaker(bind=ENGINE, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(ENGINE)
