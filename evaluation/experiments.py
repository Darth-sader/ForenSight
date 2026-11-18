"""Experiment suite (doc §27, §29-33): fragmentation, corruption, missing,
competing-fragment and metadata generalisation experiments across the sweep
scenario images, comparing conventional carving vs ForenSight stages."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from common import logger, save_json
from config import RESULTS_DIR, SCENARIOS_DIR, SEED
from evaluation.benchmarks import SignatureCarver, carve_candidates_to_sequences
from evaluation.metrics import (anomaly_metrics, classification_metrics,
                                reconstruction_metrics, topk_recovery)


def load_placements(scenario_id: str) -> List[Dict[str, object]]:
    with open(SCENARIOS_DIR / scenario_id / "placements.csv", newline="",
              encoding="utf-8") as fh:
        return [{k: (v if k in ("fragment_id", "source_file", "file_type",
                                "allocation_status", "status", "split",
                                "similar_group") else _num(v))
                 for k, v in row.items()} for row in csv.DictReader(fh)]


def _num(v):
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return v


def true_sequences_for(placements: List[Dict[str, object]]) -> List[List[str]]:
    seqs: Dict[str, List[Dict]] = {}
    for p in placements:
        if p["status"] in ("present", "decoy"):
            seqs.setdefault(str(p["source_file"]), []).append(p)
    out = []
    for fname, rows in seqs.items():
        rows.sort(key=lambda r: r["original_position"])
        out.append([str(r["fragment_id"]) for r in rows])
    return out


class ExperimentRunner:
    """Runs the doc §29-33 sweeps against a scoring/reconstruction callable."""

    def __init__(self, run_scenario_fn: Callable[[str], Dict[str, object]]):
        """``run_scenario_fn(scenario_id)`` must return a dict with keys:
        candidates (list of ReconstructionCandidate), fragments (data+metas),
        anomaly flags per fragment and transition probabilities on val pairs."""
        self.run = run_scenario_fn

    # ------------------------------------------------------------------
    def fragmentation_sweep(self, levels=("low", "medium", "severe")) -> Dict[str, object]:
        out = {}
        for lvl in levels:
            sid = f"SWEEP_FRAG_{lvl.upper()}"
            res = self.run(sid)
            out[lvl] = res["reconstruction"]
            logger.info("frag sweep %-8s placement=%.3f complete=%.3f",
                        lvl, res["reconstruction"]["fragment_placement_accuracy"],
                        res["reconstruction"]["complete_reconstruction_rate"])
        return out

    def corruption_sweep(self, levels=(1, 5, 10, 20)) -> Dict[str, object]:
        out = {}
        for pct in levels:
            sid = f"SWEEP_CORR_{pct:02d}"
            res = self.run(sid)
            out[f"{pct}%"] = {**res["reconstruction"],
                              "transition": res.get("transition", {}),
                              "confidence_mean": res.get("confidence_mean")}
            logger.info("corr sweep %d%% complete=%.3f", pct,
                        res["reconstruction"]["complete_reconstruction_rate"])
        return out

    def missing_sweep(self, levels=(5, 10, 20)) -> Dict[str, object]:
        out = {}
        for pct in levels:
            sid = f"SWEEP_MISS_{pct:02d}"
            res = self.run(sid)
            out[f"{pct}%"] = res["reconstruction"]
        return out

    def competing_experiment(self) -> Dict[str, object]:
        res = self.run("SWEEP_DECOYS")
        return res

    def metadata_experiment(self, run_metadata_aware: Callable[[str, bool], Dict],
                            scenario_ids=("EVAL_A", "EVAL_C")) -> Dict[str, object]:
        """Fragment-only vs fragment+filesystem context (doc §33)."""
        out = {}
        for sid in scenario_ids:
            with_ctx = run_metadata_aware(sid, True)
            without = run_metadata_aware(sid, False)
            out[sid] = {"with_context": with_ctx["reconstruction"],
                        "without_context": without["reconstruction"]}
        return out
