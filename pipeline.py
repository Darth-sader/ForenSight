"""ForenSight pipeline orchestrator.

``train_all``  - trains every supervised model + both deep encoders, calibrates,
                 runs CV/tuning, saves artifacts (doc §11-14, §23, §25, §26).
``analyse_image`` - full forensic analysis of a disk image (doc §4, §36):
                 extraction -> features -> embeddings -> anomaly -> graph ->
                 bidirectional island-driven inference -> constraints ->
                 ranking -> calibrated candidates + metrics + report.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from common import ReconstructionCandidate, logger, save_json, set_seeds, timer
from config import (ARTIFACTS_DIR, FRAGMENT_SIZE, RESULTS_DIR, SCORE_WEIGHTS, SEED,
                    SUPERBLOCK_BLOCKS)
from calibration.confidence import fit_best_calibrator
from embeddings import FragmentEmbedder, FusedEmbedder
from models import (LogisticModel, MLPTorchModel, RandomForestModel,
                    TransitionEnsemble, XGBoostModel, build_tabular_models)
from models.cnn_encoder import SiameseCNN
from features import FeatureEngine
from preprocessing.labels import CorpusPairBuilder
from reconstruction.reconstruct import ReconstructionEngine
from ranking.candidate_ranker import CandidateRanker, candidate_evidence_row


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------

@dataclass
class Artifacts:
    dir: Path
    same_ensemble: TransitionEnsemble
    next_ensemble: TransitionEnsemble
    next_calibrator: object
    same_calibrator: object
    fused_embedder: FusedEmbedder
    ranker: CandidateRanker
    confidence_calibrator: object
    file_size_stats: Dict[str, Dict[str, float]]
    training_summary: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def load(directory: Path = ARTIFACTS_DIR) -> "Artifacts":
        import joblib

        d = Path(directory)
        same = joblib.load(d / "same_ensemble.joblib")
        nxt = joblib.load(d / "next_ensemble.joblib")
        ncal = joblib.load(d / "next_calibrator.joblib")
        scal = joblib.load(d / "same_calibrator.joblib")
        ranker = joblib.load(d / "ranker.joblib")
        ccal = joblib.load(d / "confidence_calibrator.joblib")
        stats = joblib.load(d / "file_size_stats.joblib")

        mlp_enc = FragmentEmbedder(input_dim=1)
        mlp_enc.load(d / "encoder_mlp.pt")
        cnn = SiameseCNN()
        cnn.load(d / "encoder_cnn.pt")
        fused = FusedEmbedder({"handcrafted": mlp_enc, "raw": cnn})
        return Artifacts(d, same, nxt, ncal, scal, fused, ranker, ccal, stats)


def _file_size_stats() -> Dict[str, Dict[str, float]]:
    from collections import defaultdict

    rows = list(csv.DictReader(open(Path(__file__).parent / "dataset" / "ground_truth" / "files.csv",
                                    encoding="utf-8")))
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["file_type"]].append(int(r["size_bytes"]))
    out = {}
    for t, sizes in by_type.items():
        a = np.array(sizes)
        out[t] = {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
                  "p90": float(np.percentile(a, 90))}
    return out


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_all(tune: bool = True, cv: bool = True, seed: int = SEED) -> Artifacts:
    """Train the complete ML stack and persist artifacts (doc §11-26)."""
    set_seeds(seed)
    t_start = time.time()
    builder = CorpusPairBuilder(seed=seed)
    data = builder.build()
    train_ds, val_ds, hold_ds = data["train_ds"], data["val_ds"], data["holdout_ds"]
    engine: FeatureEngine = data["engine"]
    logger.info("pairs: train=%d val=%d holdout=%d | dims=%d",
                len(train_ds), len(val_ds), len(hold_ds), train_ds.X.shape[1])
    summary: Dict[str, object] = {"n_train_pairs": len(train_ds), "n_val_pairs": len(val_ds),
                                  "n_holdout_pairs": len(hold_ds),
                                  "n_features": int(train_ds.X.shape[1])}

    # ------------------------------------------------ supervised models (§11-12)
    ensembles = {}
    calibrators = {}
    for task, y_tr, y_va, y_ho in (("same", train_ds.y_same, val_ds.y_same, hold_ds.y_same),
                                   ("next", train_ds.y_next, val_ds.y_next, hold_ds.y_next)):
        ens = TransitionEnsemble()
        for model in build_tabular_models(seed=seed):
            name = model.name
            model.fit(train_ds.X, y_tr, val_ds.X, y_va)
            val_auc = _safe_auc(y_va, model.predict_proba(val_ds.X))
            hold = _safe_auc(y_ho, model.predict_proba(hold_ds.X))
            ens.add(name, model, val_auc)
            logger.info("[%s] %-20s val_auc=%.4f holdout_auc=%.4f", task, name, val_auc, hold)
            summary[f"{task}_{name}_val_auc"] = val_auc
            summary[f"{task}_{name}_holdout_auc"] = hold
        ens.finalize()
        ensembles[task] = ens

        # probability calibration on validation data (doc §23)
        raw_val = ens.predict_proba(val_ds.X)
        cal = fit_best_calibrator(raw_val, y_va)
        calibrators[task] = cal
        from calibration.confidence import brier_score, expected_calibration_error
        cal_val = cal.transform(raw_val)
        summary[f"{task}_brier_raw"] = brier_score(raw_val, y_va)
        summary[f"{task}_brier_calibrated"] = brier_score(cal_val, y_va)
        summary[f"{task}_ece_raw"] = expected_calibration_error(raw_val, y_va)
        summary[f"{task}_ece_calibrated"] = expected_calibration_error(cal_val, y_va)
        logger.info("[%s] calibrator=%s brier %.4f -> %.4f", task, cal.name,
                    summary[f"{task}_brier_raw"], summary[f"{task}_brier_calibrated"])

    # ------------------------------------------------ hyperparameter search (§25)
    if tune:
        from optimization import tune_model
        tuned: Dict[str, object] = {}
        for family, task, y_tr, y_va in (("random_forest", "next", train_ds.y_next, val_ds.y_next),
                                         ("xgboost", "next", train_ds.y_next, val_ds.y_next),
                                         ("mlp_torch", "next", train_ds.y_next, val_ds.y_next)):
            strategy = "bayesian" if family != "random_forest" else "grid"
            best_params, best_auc, log = tune_model(family, train_ds.X, y_tr, val_ds.X, y_va,
                                                    strategy=strategy, n_trials=5, seed=seed)
            tuned[family] = {"strategy": strategy, "best_params": best_params,
                             "best_val_auc": best_auc, "log": log}
            logger.info("[tune] %s best_auc=%.4f %s", family, best_auc, best_params)
            # refit the winning configuration
            model = _refit(family, best_params, seed, train_ds, y_tr, val_ds, y_va)
            ens = ensembles[task]
            val_auc = _safe_auc(y_va, model.predict_proba(val_ds.X))
            if val_auc > ens.val_auc.get(model.name, 0.0):
                ens.add(model.name, model, val_auc)
                ens.finalize()
        summary["hyperparameter_search"] = tuned

    # ------------------------------------------------ cross-validation (§26)
    if cv:
        from evaluation.cross_validation import group_kfold_by_file, no_leakage_check
        assert no_leakage_check(train_ds.group, hold_ds.group)
        cv_res = group_kfold_by_file(train_ds.X, train_ds.y_next, train_ds.group,
                                     lambda: XGBoostModel(seed=seed), k=3)
        summary["cv_xgboost_next"] = cv_res["aggregate"]
        logger.info("[cv] group k-fold xgb: auc=%.4f±%.4f",
                    cv_res["aggregate"]["roc_auc"], cv_res["aggregate"]["roc_auc_std"])

    # ------------------------------------------------ deep representations (§13-14)
    # NOTE: embedders receive the FULL engine matrix; pair indices are engine
    # row indices (train pairs only reference train rows inside it).
    with timer("train MLP encoder (handcrafted siamese)", logger):
        mlp_embedder = FragmentEmbedder(input_dim=engine.single_matrix_.shape[1])
        mlp_embedder.fit(engine.single_matrix_, train_ds.pair_idx,
                         train_ds.y_same, train_ds.y_next,
                         X_val_single=engine.single_matrix_,
                         val_pairs=val_ds.pair_idx, val_labels_same=val_ds.y_same)
    with timer("train raw-byte CNN encoder (siamese)", logger):
        cnn_embedder = SiameseCNN(block_size=FRAGMENT_SIZE, seed=seed)
        cnn_embedder.fit(engine.data_, train_ds.pair_idx,
                         train_ds.y_same, train_ds.y_next,
                         val_raw=engine.data_,
                         val_pairs=val_ds.pair_idx, val_labels_same=val_ds.y_same)
    fused = FusedEmbedder({"handcrafted": mlp_embedder, "raw": cnn_embedder})

    # embedding quality probe: holdout pair separation
    emb_all = fused.embed_all({"handcrafted": engine.single_matrix_, "raw": engine.data_})
    from embeddings.similarity import pair_similarities
    same_mask = hold_ds.y_same == 1
    diff_mask = ~same_mask
    summary["embedding_separation_mlp"] = _separation(
        pair_similarities(emb_all["handcrafted"], hold_ds.pair_idx), same_mask, diff_mask)
    summary["embedding_separation_cnn"] = _separation(
        pair_similarities(emb_all["raw"], hold_ds.pair_idx), same_mask, diff_mask)

    # ------------------------------------------------ candidate ranker (§21)
    ranker, conf_cal = _train_ranker(data, ensembles["next"], calibrators["next"], fused, seed)

    artifacts = Artifacts(ARTIFACTS_DIR, ensembles["same"], ensembles["next"],
                          calibrators["next"], calibrators["same"], fused,
                          ranker, conf_cal, _file_size_stats(), summary)
    _save_artifacts(artifacts)
    summary["training_seconds"] = round(time.time() - t_start, 1)
    save_json(RESULTS_DIR / "training_summary.json", summary)
    logger.info("training complete in %.1fs", summary["training_seconds"])
    return artifacts


def _train_rows(data: dict) -> np.ndarray:
    """Engine rows belonging to training files (originals + corruption variants)."""
    metas = data["metas"]
    return np.array([i for i, m in enumerate(metas) if m["split"] == "train"], dtype=np.int64)


def _separation(sims: np.ndarray, same_mask: np.ndarray, diff_mask: np.ndarray) -> Dict[str, float]:
    return {"same_file_mean_sim": float(np.mean(sims[same_mask])),
            "different_file_mean_sim": float(np.mean(sims[diff_mask])),
            "margin": float(np.mean(sims[same_mask]) - np.mean(sims[diff_mask]))}


def _safe_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return 0.5


def _refit(family, params, seed, train_ds, y_tr, val_ds, y_va):
    if family == "random_forest":
        m = RandomForestModel(seed=seed, **params)
    elif family == "xgboost":
        m = XGBoostModel(seed=seed, **params)
    else:
        m = MLPTorchModel(seed=seed, **params)
    m.fit(train_ds.X, y_tr, val_ds.X, y_va)
    return m


def _train_ranker(data, next_ens, next_cal, fused, seed) -> Tuple[CandidateRanker, object]:
    """Learning-to-rank meta-model trained on candidates from training images."""
    from evaluation.experiments import load_placements

    ranker = CandidateRanker()
    hyps_all, labels_all = [], []
    for scenario_id in ("S_A", "S_C"):
        img = Path(__file__).parent / "dataset" / "disk_images" / f"{scenario_id}.img"
        if not img.exists():
            continue
        blk_map = {}
        try:
            for p in load_placements(scenario_id):
                if p["status"] == "present":
                    blk_map[int(p["cluster"])] = (str(p["source_file"]),
                                                  int(p["original_position"]))
        except FileNotFoundError:
            continue
        result = analyse_image(img, name=f"train_{scenario_id}", artifacts=None,
                               _override=(data, next_ens, next_cal, fused, seed))
        hyps = result["hypotheses"]
        for h in hyps:
            seq = [blk_map.get(int(n)) for n in h["path"].nodes]
            if any(s is None for s in seq):
                labels_all.append(0)
                hyps_all.append(h)
                continue
            files = {s[0] for s in seq}
            positions = [s[1] for s in seq]
            # correct candidate: single source file, positions strictly increasing
            correct = len(files) == 1 and all(
                positions[i] < positions[i + 1] for i in range(len(positions) - 1))
            hyps_all.append(h)
            labels_all.append(int(correct))
    if len(set(labels_all)) >= 2:
        ranker.fit_ltr(hyps_all, labels_all)
        scores = np.array([ranker._score(h) for h in hyps_all])
        conf_cal = fit_best_calibrator(scores, np.array(labels_all))
        logger.info("ranker trained on %d candidates (%.0f%% correct)",
                    len(labels_all), 100 * np.mean(labels_all))
    else:
        conf_cal = None
    return ranker, conf_cal


def _save_artifacts(a: Artifacts) -> None:
    import joblib

    d = a.dir
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(a.same_ensemble, d / "same_ensemble.joblib")
    joblib.dump(a.next_ensemble, d / "next_ensemble.joblib")
    joblib.dump(a.next_calibrator, d / "next_calibrator.joblib")
    joblib.dump(a.same_calibrator, d / "same_calibrator.joblib")
    joblib.dump(a.ranker, d / "ranker.joblib")
    joblib.dump(a.confidence_calibrator, d / "confidence_calibrator.joblib")
    joblib.dump(a.file_size_stats, d / "file_size_stats.joblib")
    a.fused_embedder.embedders["handcrafted"].save(d / "encoder_mlp.pt")
    a.fused_embedder.embedders["raw"].save(d / "encoder_cnn.pt")
    logger.info("artifacts saved to %s", d)


# --------------------------------------------------------------------------
# Image analysis (doc §4 pipeline, §36 output)
# --------------------------------------------------------------------------

def analyse_image(image_path: Path, name: str, artifacts: Optional[Artifacts] = None,
                  use_metadata: bool = True, seed: int = SEED,
                  mode: str = "bidirectional_island",
                  _override: Optional[tuple] = None) -> Dict[str, object]:
    """Analyse one disk image end-to-end and return the full result bundle."""
    from extraction.fragment_extractor import extract_fragments

    set_seeds(seed)
    if _override is not None:
        data, next_ens, next_cal, fused, seed = _override
        ranker, conf_cal, size_stats = CandidateRanker(), None, _file_size_stats()
    else:
        if artifacts is None:
            artifacts = Artifacts.load()
        next_ens, next_cal = artifacts.next_ensemble, artifacts.next_calibrator
        fused = artifacts.fused_embedder
        ranker, conf_cal, size_stats = artifacts.ranker, artifacts.confidence_calibrator, \
            artifacts.file_size_stats

    t0 = time.time()
    fragments, byte_data, image_summary = extract_fragments(image_path, FRAGMENT_SIZE, seed=seed)
    metas = [f.to_dict() for f in fragments]
    if not use_metadata:  # metadata-independent condition (doc §33)
        for m in metas:
            m["allocation_status"] = "raw"
            m["filesystem_context"] = {"fs_type": "", "volume_id": "", "neighbors": []}
            m["physical_offset"] = 0.0
            m["cluster"] = 0

    engine = FeatureEngine(byte_data, metas)
    n = engine.n_
    logger.info("[%s] fragments=%d features=%d", name, n, engine.single_matrix_.shape[1])

    # unsupervised layers (doc §16-17)
    embeddings = fused.embed_all({"handcrafted": engine.single_matrix_, "raw": engine.data_})
    emb_avg = np.mean(list(embeddings.values()), axis=0)

    from anomaly import IsolationForestDetector, LOFDetector, anomaly_status, combine_scores
    if_detector = IsolationForestDetector(seed=seed).fit(emb_avg)
    lof_detector = LOFDetector().fit(emb_avg)
    anomaly_scores = combine_scores([if_detector.anomaly_scores(emb_avg),
                                     lof_detector.anomaly_scores(emb_avg)])
    anomaly_flags = anomaly_scores >= 0.7

    from clustering import KMeansClusterer, cluster_summary
    kmeans = KMeansClusterer().fit(emb_avg)
    cluster_labels = kmeans.predict(emb_avg) if kmeans.model is not None else np.zeros(n, dtype=int)

    # reconstruction (doc §19-22)
    recon = ReconstructionEngine(engine, next_ens, fused, next_cal, anomaly_scores,
                                 file_size_stats=size_stats,
                                 image_size_blocks=image_summary["total_blocks"])
    hypotheses = recon.analyse_image([f.fragment_id for f in fragments],
                                     mode=mode,
                                     file_type_of=lambda i: _header_type(engine, i))
    candidates = ranker.rank(
        hypotheses,
        fragment_id_of=lambda i: fragments[i].fragment_id,
        file_type_of=lambda i: _header_type(engine, i),
        anomaly_of={i: float(anomaly_scores[i]) for i in range(n)},
        calibrated_confidence_of=(lambda h: float(conf_cal.transform(np.array([ranker._score(h)]))[0])
                                  if conf_cal is not None else None))

    # ------------------------------------------------ evaluation vs ground truth
    metrics: Dict[str, object] = {}
    truth = _truth_for(image_path.name)
    if truth is not None:
        from evaluation.metrics import (reconstruction_metrics, topk_recovery)
        from evaluation.benchmarks import SignatureCarver, benchmark_components, \
            carve_candidates_to_sequences
        true_seqs, gap_truth, blk2canon = truth
        # predictions use block-based ids; translate to canonical fragment ids
        pred_seqs = [[blk2canon.get(f, f) for f in c.fragment_sequence]
                     for c in candidates[:len(true_seqs) * 3]]
        metrics["reconstruction"] = reconstruction_metrics(true_seqs, pred_seqs, gap_truth)
        metrics["topk"] = topk_recovery(true_seqs, pred_seqs)
        carver = SignatureCarver(FRAGMENT_SIZE)
        carved = carver.carve(engine.data_, metas)
        carve_seqs = [[blk2canon.get(fragments[i].fragment_id, fragments[i].fragment_id)
                       for i in c["sequence"]] for c in carved]
        metrics["carving_baseline"] = reconstruction_metrics(true_seqs, carve_seqs)
        metrics["components"] = benchmark_components(
            None, true_seqs, None, full_sequences=pred_seqs)
        metrics["top1_carving"] = topk_recovery(true_seqs, carve_seqs, ks=(1,))

    result = {
        "name": name,
        "image": str(image_path),
        "image_summary": image_summary,
        "fragments": [_fragment_record(i, engine, fragments[i], cluster_labels,
                                       anomaly_scores, anomaly_flags, emb_avg) for i in range(n)],
        "graph": _graph_record(recon.graph_, fragments, anomaly_scores),
        "clusters": {"labels": cluster_labels.tolist(), "k": kmeans.k_,
                     "silhouette": kmeans.silhouette_,
                     "summary": cluster_summary(cluster_labels, engine.metas_)},
        "anomaly": {"scores": anomaly_scores.tolist(),
                    "flagged": int(anomaly_flags.sum()),
                    "detectors": ["isolation_forest", "local_outlier_factor"]},
        "candidates": candidates,
        "hypotheses": hypotheses,
        "metrics": metrics,
        "reconstruction_mode": mode,
        "seconds": round(time.time() - t0, 1),
    }
    logger.info("[%s] analysis done: %d candidates in %.1fs", name, len(candidates),
                result["seconds"])
    return result


def _header_type(engine: FeatureEngine, i: int) -> Optional[str]:
    s = engine.struct_dict(i)
    types = ("JPEG", "PNG", "PDF", "ZIP", "DOCX", "XLSX", "MP3", "MP4")
    best, best_score = None, 0.5
    for t in types:
        if s[f"header_{t}"] > best_score:
            best, best_score = t, s[f"header_{t}"]
    return best


def _fragment_record(i, engine, frag, cluster_labels, anomaly_scores, anomaly_flags,
                     emb_avg) -> Dict[str, object]:
    s = engine.struct_dict(i)
    return {
        "fragment_id": frag.fragment_id,
        "index": i,
        "source_file": frag.source_file,          # None for image analyses (unknown)
        "file_type": frag.file_type or _header_type(engine, i) or "unknown",
        "original_position": frag.original_position,
        "fragment_size": frag.fragment_size,
        "physical_offset": frag.physical_offset,
        "cluster": frag.cluster,
        "allocation_status": frag.allocation_status,
        "fs_cluster_label": int(cluster_labels[i]),
        "anomaly_score": round(float(anomaly_scores[i]), 4),
        "anomaly_status": "SUSPICIOUS" if anomaly_flags[i] else "NORMAL",
        "entropy": round(float(engine.cache_["byte"]["entropy"][i]), 4),
        "header_evidence": round(float(s["header_detected"]), 3),
        "footer_evidence": round(float(s["footer_detected"]), 3),
        "format_consistency": round(float(s["format_consistency"]), 3),
        "embedding_preview": [round(float(x), 3) for x in emb_avg[i][:8]],
    }


def _graph_record(graph, fragments, anomaly_scores) -> Dict[str, object]:
    from visualization.graphs import graph_to_frontend
    rec = graph_to_frontend(graph, lambda i: fragments[i].fragment_id,
                            {i: float(anomaly_scores[i]) for i in range(graph.number_of_nodes())})
    return rec


def _truth_for(image_filename: str) -> Optional[Tuple[List[List[str]], Dict[str, List[int]], Dict[str, str]]]:
    """Ground truth for generated scenario images: true sequences, expected gap
    positions, and a block-index -> canonical fragment id map (None for uploads)."""
    from evaluation.experiments import load_placements, true_sequences_for

    scenario = Path(image_filename).stem
    p = Path(__file__).parent / "dataset" / "scenarios" / scenario / "placements.csv"
    if not p.exists():
        return None
    placements = load_placements(scenario)
    seqs = true_sequences_for(placements)
    seqs = [s for s in seqs if s]
    gap_truth: Dict[str, List[int]] = {}
    blk2canon: Dict[str, str] = {}
    for p_row in placements:
        if p_row["status"] == "missing":
            gap_truth.setdefault(str(p_row["source_file"]), []).append(
                int(p_row["original_position"]))
        elif p_row["status"] == "present":
            blk2canon[f"F{int(p_row['cluster']):05d}"] = str(p_row["fragment_id"])
    return seqs, gap_truth, blk2canon


if __name__ == "__main__":
    import sys
    tune = "--no-tune" not in sys.argv
    cv = "--no-cv" not in sys.argv
    train_all(tune=tune, cv=cv)

