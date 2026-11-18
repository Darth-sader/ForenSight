"""End-to-end reconstruction engine (doc §4, §19-22, §36 + extensions).

Takes a forensic disk image, extracts fragments, scores candidate successor
pairs with the trained ML ensemble, builds the relationship graph, and runs
**bidirectional, island-driven** beam-search inference with forensic
constraints, emitting ranked, calibrated reconstruction candidates:

* forward beam search from header-fragment seeds
* backward beam search from footer-fragment seeds (bidirectional, stitched)
* island partitioning of the high-confidence graph restricts the search space
  per connected component (island-driven inference)
* [UNKNOWN] gaps are emitted instead of fabricated fragments (doc §31)
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from common import logger, set_seeds
from config import (BEAM_WIDTH, FRAGMENT_SIZE, GRAPH_TOP_K_SUCCESSORS,
                    MAX_CANDIDATES_PER_SEED, SCORE_WEIGHTS, SEED)
from features import FeatureEngine
from graph.graph_inference import (CandidatePath, backward_beam_search,
                                   beam_search, island_partition,
                                   stitch_bidirectional)
from graph.relationship_graph import build_relationship_graph, graph_stats
from reconstruction.constraints import ForensicConstraints


class ReconstructionEngine:
    """ML inference + graph inference + constraint validation (doc §19-22)."""

    def __init__(self, engine: FeatureEngine, transition_ensemble,
                 embedder, calibrator, anomaly_scores: np.ndarray,
                 file_size_stats: Optional[Dict[str, Dict[str, float]]] = None,
                 image_size_blocks: int = 0,
                 island_threshold: float = 0.5):
        self.engine = engine
        self.ensemble = transition_ensemble
        self.embedder = embedder          # FusedEmbedder-like (embed_all) or simple (.embed)
        self.calibrator = calibrator
        self.anomaly = np.asarray(anomaly_scores, dtype=np.float64)
        self.island_threshold = island_threshold
        self.constraints = ForensicConstraints(engine, FRAGMENT_SIZE,
                                               file_size_stats, image_size_blocks)

    # ------------------------------------------------------------------ pairs
    def _score_all_pairs(self, candidate_pairs: np.ndarray) -> Dict[str, np.ndarray]:
        """Score a set of (a, b) successor candidates with every evidence source."""
        X = self.engine.pair_matrix_vec(candidate_pairs)
        raw = self.ensemble.predict_proba(X)
        calibrated = (self.calibrator.transform(raw)
                      if self.calibrator is not None else raw)
        if self.embeddings_:
            emb_sim = self._pair_embedding_similarity(candidate_pairs)
        else:
            emb_sim = np.zeros(len(candidate_pairs))
        structural = self._structural_compat_vec(candidate_pairs)
        context = self._context_compat_vec(candidate_pairs)
        return {"raw": raw, "calibrated": calibrated, "emb_sim": emb_sim,
                "structural": structural, "context": context}

    def _pair_embedding_similarity(self, pairs: np.ndarray) -> np.ndarray:
        sims = None
        for emb in self.embeddings_.values():
            ea, eb = emb[pairs[:, 0]], emb[pairs[:, 1]]
            den = np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1) + 1e-9
            s = np.clip((ea * eb).sum(axis=1) / den, -1, 1)
            sims = s if sims is None else sims + s
        return sims / max(1, len(self.embeddings_))

    def _structural_compat_vec(self, pairs: np.ndarray) -> np.ndarray:
        from features.pair_features import structural_pair_features
        ia, ib = pairs[:, 0], pairs[:, 1]
        a_list = [self.engine.struct_dict(int(i)) for i in ia]
        b_list = [self.engine.struct_dict(int(j)) for j in ib]
        out = np.empty(len(pairs))
        for k, (a_s, b_s) in enumerate(zip(a_list, b_list)):
            f = structural_pair_features(a_s, b_s)
            out[k] = 0.5 * f["struct_same_format_detected"] + \
                0.2 * f["struct_header_then_body"] + 0.3 * f["struct_consistency_min"]
        return out

    def _context_compat_vec(self, pairs: np.ndarray) -> np.ndarray:
        from features.pair_features import context_pair_features
        ia, ib = pairs[:, 0], pairs[:, 1]
        block = float(self.engine.data_.shape[1])
        out = np.empty(len(pairs))
        for k, (i, j) in enumerate(zip(ia, ib)):
            f = context_pair_features(self.engine.metas_[int(i)], self.engine.metas_[int(j)], block)
            out[k] = f["context_compat"]
        return out

    # ------------------------------------------------------------------ main
    def analyse_image(self, fragment_ids: List[str],
                      successors_per_fragment: int = 8,
                      forbidden: Optional[set] = None,
                      top_k_graph: int = GRAPH_TOP_K_SUCCESSORS,
                      max_candidates_per_seed: int = MAX_CANDIDATES_PER_SEED,
                      max_seeds: int = 60,
                      mode: str = "bidirectional_island",
                      file_type_of: Optional[Callable[[int], str]] = None) -> List[dict]:
        """Graph inference + constraint validation (doc §19 + extensions).

        ``mode``: "forward" | "bidirectional" | "bidirectional_island".
        Returns hypothesis dicts consumed by the candidate ranker.
        """
        set_seeds(SEED)
        n = self.engine.n_
        self.embeddings_ = self._compute_embeddings()

        # 1. candidate successor pairs: fused-embedding KNN union + physical
        #    neighbours + controlled random probes (pair generation, doc §8)
        rng = random.Random(SEED)
        pairs: List[Tuple[int, int]] = []
        if self.embeddings_:
            from embeddings import FusedEmbedder
            knn_idx, _ = FusedEmbedder.knn_union(self.embeddings_,
                                                 successors_per_fragment, cap=successors_per_fragment + 4)
        else:
            knn_idx = None
        for a in range(n):
            cand: set = set()
            if knn_idx is not None:
                cand.update(int(x) for x in knn_idx[a] if x != a)
            cand.update([(a + 1) % n, (a + 2) % n])
            cand.discard(a)
            for _ in range(2):
                cand.add(rng.randrange(n))
            for b in cand:
                pairs.append((a, b))
        pair_arr = np.array(pairs, dtype=np.int64)
        scores = self._score_all_pairs(pair_arr)

        # 2. relationship graph (doc §18)
        graph = build_relationship_graph(
            fragment_ids, pair_arr, scores["calibrated"], scores["calibrated"],
            scores["emb_sim"], scores["structural"], scores["context"],
            self.anomaly, top_k=top_k_graph)
        self.graph_ = graph
        self.graph_summary_ = graph_stats(graph)

        # 3. seed selection: headers (forward) + footers (backward)
        header_seeds, footer_seeds = self._select_seeds(max_seeds)

        def constraints_fn(path):
            ftype = file_type_of(path.nodes[0]) if file_type_of else None
            return self.constraints.check(path, ftype, used_elsewhere=forbidden)

        # 4. graph inference per mode (doc §19 + bidirectional/island extension)
        paths = self._infer_paths(graph, header_seeds, footer_seeds,
                                  forbidden, mode)
        self.paths_ = paths

        # 5. component scoring + constraint validation (doc §20, §22)
        from graph.path_optimizer import path_components
        hypotheses: List[dict] = []
        for rank, path in enumerate(paths):
            violations, issues, evidence = constraints_fn(path)
            components = path_components(path)
            from graph.path_optimizer import combined_score
            raw = combined_score(components, violations)
            hypotheses.append({"seed": path.seed, "rank_in_seed": rank, "path": path,
                               "violations": violations, "issues": issues,
                               "components": components, "raw_score": float(raw),
                               "evidence": evidence, "pair_scores": scores})
        self.scores_ = scores
        return hypotheses

    # ------------------------------------------------------------------
    def _compute_embeddings(self) -> Dict[str, np.ndarray]:
        if self.embedder is None:
            return {}
        if hasattr(self.embedder, "embed_all"):
            return self.embedder.embed_all({"handcrafted": self.engine.single_matrix_,
                                            "raw": self.engine.data_})
        return {"handcrafted": self.embedder.embed(self.engine.single_matrix_)}

    def _select_seeds(self, max_seeds: int) -> Tuple[List[int], List[int]]:
        n = self.engine.n_
        struct = self.engine.cache_["structural"]
        header_seeds = [i for i in range(n) if struct["header_detected"][i] >= 1.0]
        header_seeds.sort(key=lambda i: -max(
            struct[f"header_{t}"][i] for t in
            ("JPEG", "PNG", "PDF", "ZIP", "DOCX", "XLSX", "MP3", "MP4")))
        footer_seeds = [i for i in range(n) if struct["is_footer_fragment"][i] >= 1.0]
        return header_seeds[:max_seeds], footer_seeds[:max_seeds]

    def _infer_paths(self, graph, header_seeds, footer_seeds, forbidden, mode) -> List[CandidatePath]:
        paths: List[CandidatePath] = []
        if mode == "forward":
            for s in header_seeds:
                paths.extend(beam_search(graph, s, forbidden=forbidden))
            return paths

        bwd_kwargs = dict(forbidden=forbidden)
        if mode == "bidirectional":
            for s in header_seeds:
                paths.extend(beam_search(graph, s, forbidden=forbidden))
            for s in footer_seeds:
                paths.extend(backward_beam_search(graph, s, **bwd_kwargs))
            paths.extend(stitch_bidirectional(
                [p for p in paths if p.direction == "forward"],
                [p for p in paths if p.direction == "backward"]))
            return self._dedup_paths(paths)

        # bidirectional_island (default): partition into islands first
        islands = island_partition(graph, self.island_threshold)
        logger.info("island partition: %d islands from %d nodes",
                    len(islands), graph.number_of_nodes())
        assigned = set().union(*islands) if islands else set()
        leftovers = [n for n in graph.nodes() if n not in assigned]
        for island in islands:
            h = [s for s in header_seeds if s in island]
            f = [s for s in footer_seeds if s in island]
            if not h and not f:
                # no structural seeds: grow from the island's best-connected node
                best = max(island, key=lambda x: graph.degree(x))
                h, f = [best], [best]
            for s in h:
                paths.extend(beam_search(graph, s, forbidden=forbidden, allowed=island))
            for s in f:
                paths.extend(backward_beam_search(graph, s, allowed=island, **bwd_kwargs))
            paths.extend(stitch_bidirectional(
                [p for p in paths if p.direction == "forward" and p.nodes[-1] in island],
                [p for p in paths if p.direction == "backward" and p.nodes[0] in island]))
        # unassigned fragments stay recoverable via later orphan-cluster handling
        return self._dedup_paths(paths)

    @staticmethod
    def _dedup_paths(paths: List[CandidatePath]) -> List[CandidatePath]:
        seen: Dict[tuple, CandidatePath] = {}
        for p in sorted(paths, key=lambda p: -p.avg_score):
            seen.setdefault(tuple(p.nodes), p)
        return list(seen.values())
