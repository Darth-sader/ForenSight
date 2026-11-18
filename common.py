"""Shared utilities and data structures used across the ForenSight pipeline."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config import SEED, ensure_dirs


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    except Exception:
        pass


def timer(name: str, log: Optional[logging.Logger] = None) -> Any:
    """Context manager that logs the wall time of a block."""
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *exc):
            dt = time.time() - self.t0
            msg = f"{name}: {dt:.2f}s"
            (log or logger).info(msg)
            print(f"[forensight] {msg}")
            return False

    return _T()


def get_logger(name: str = "forensight") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to plain JSON-serialisable python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path | str, obj: Any) -> None:
    ensure_dirs()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Core forensic record structures (doc §6)
# --------------------------------------------------------------------------

@dataclass
class Fragment:
    """One recovered storage block with its forensic metadata."""
    fragment_id: str
    fragment_size: int
    physical_offset: int          # byte offset inside the disk image
    cluster: int                  # filesystem cluster index
    allocation_status: str        # "allocated" | "unallocated" | "system"
    source_file: Optional[str] = None      # ground truth only; None = unknown
    file_type: Optional[str] = None        # ground truth only
    original_position: Optional[int] = None  # ground truth only
    corruption_level: float = 0.0
    status: str = "present"       # present | missing | decoy
    filesystem_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairSample:
    """A candidate fragment pair with supervised labels (doc §11)."""
    fragment_a: str
    fragment_b: str
    label_same: int               # 1 = same file
    label_next: int               # 1 = B immediately follows A
    hard_negative: bool = False


@dataclass
class ReconstructionCandidate:
    """A ranked reconstruction hypothesis (doc §36)."""
    candidate_id: str
    file_type: str
    fragment_sequence: List[str]
    estimated_size: int
    confidence: float                 # calibrated confidence 0..1
    raw_score: float
    classification: str               # HIGH-CONFIDENCE RECOVERY / UNCERTAIN / ...
    ml_evidence: Dict[str, Any] = field(default_factory=dict)
    forensic_evidence: Dict[str, Any] = field(default_factory=dict)
    anomaly_analysis: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    gaps: List[int] = field(default_factory=list)   # indices of [UNKNOWN] gaps

    def sequence_string(self) -> str:
        parts: List[str] = []
        for i, fid in enumerate(self.fragment_sequence):
            parts.append(fid)
            if i in self.gaps:
                parts.append("[UNKNOWN]")
        return " -> ".join(parts)


def balanced_subsample_indices(labels: np.ndarray, rng: random.Random,
                               max_per_class: Optional[int] = None) -> np.ndarray:
    """Return indices balanced across classes (controlled sampling, doc §8)."""
    idx_by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        idx_by_class.setdefault(int(y), []).append(i)
    n = min(len(v) for v in idx_by_class.values())
    if max_per_class is not None:
        n = min(n, max_per_class)
    picked: List[int] = []
    for cls, idx in sorted(idx_by_class.items()):
        rng.shuffle(idx)
        picked.extend(idx[:n])
    rng.shuffle(picked)
    return np.array(sorted(picked))
