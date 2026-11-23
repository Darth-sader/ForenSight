"""Global Constrained Forensic Reconstruction Optimizer (v2.0.0-phase5).

Integrates:
1. Mathematical decision variables: x_ij (direct), g_ij (gap), s_i (head), t_i (tail), u_i (MTZ order)
2. Clamped logit evidence objective with gap penalties and file count regularization
3. Linear constraints: in/out degree bounds, flow conservation, MTZ subtour elimination,
   format boundaries, and incompatible pair exclusions
4. HiGHS MILP and Google OR-Tools CP-SAT discrete solvers
5. Path tracing and ReconstructedFile artifact generation with gap placeholders
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

# Anchor project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase1_leakage_free.fragment_schema_v2 import ObservableFragment
from phase2_dataset_evaluation.evaluation_engine import ReconstructedFile
from phase5_constrained_optimization.cp_sat_solver import CPSATReconstructionSolver
from phase5_constrained_optimization.format_constraints import (
    FormatConstraintExtractor,
    FormatEvidence,
)
from phase5_constrained_optimization.milp_solver import (
    MILPReconstructionSolver,
    MILPSolverResult,
)
from phase5_constrained_optimization.missing_fragment_model import (
    GapEdgeInfo,
    MissingFragmentModel,
)
from phase5_constrained_optimization.optimization_constraints import ConstraintMatrixBuilder
from phase5_constrained_optimization.optimization_variables import (
    OptimizationVariableBuilder,
    VariableIndexMap,
)


@dataclass
class OptimizerConfig:
    """Hyperparameters and operational configuration for the global optimizer."""
    time_limit_seconds: float = 60.0
    mip_rel_gap: float = 1e-4
    solver_type: str = "milp"               # "milp" (HiGHS) or "cp_sat" (OR-Tools)
    enable_mtz_ordering: bool = True       # MTZ cycle / subtour elimination
    enable_gaps: bool = True               # Missing fragment gap modeling
    enable_format_constraints: bool = True # Hard format boundary and exclusion guards
    file_penalty: float = 1.0              # Regularization penalty lambda per reconstructed file
    head_bonus: float = 2.0                # Objective bonus for detected format headers
    tail_bonus: float = 2.0                # Objective bonus for detected format footers
    base_gap_penalty: float = 2.0          # Base penalty gamma_0 for missing fragment transitions
    logit_clamp: float = 10.0              # Clamp bounds [-logit_clamp, logit_clamp]
    min_prob_clamp: float = 1e-4           # Extreme probability clamp
    tau_same_min: float = 0.20             # Threshold below which fragments cannot transition
    tau_prior: float = 0.08                # Noise floor prior probability for log-likelihood ratio


@dataclass
class OptimizationOutput:
    """Comprehensive output container for the global reconstruction optimizer."""
    reconstructed_files: List[ReconstructedFile]
    solver_result: MILPSolverResult
    num_nodes: int
    num_candidate_edges: int
    num_gap_edges: int
    num_constraints: int
    num_variables: int
    runtime_seconds: float
    format_evidence: Dict[int, FormatEvidence]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class GlobalConstrainedReconstructionOptimizer:
    """Solves the discrete multi-file forensic reconstruction problem via constrained optimization."""

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.format_extractor = FormatConstraintExtractor(
            head_bonus=self.config.head_bonus,
            tail_bonus=self.config.tail_bonus,
        )
        self.gap_model = MissingFragmentModel(
            base_gap_penalty=self.config.base_gap_penalty,
        )

    def _prob_to_logit(self, p: float, prior: Optional[float] = None) -> float:
        """Convert probability to numerically stable clamped log-likelihood ratio relative to prior."""
        p_clamped = min(max(p, self.config.min_prob_clamp), 1.0 - self.config.min_prob_clamp)
        raw_logit = math.log(p_clamped / (1.0 - p_clamped))
        if prior is not None:
            prior_clamped = min(max(prior, self.config.min_prob_clamp), 1.0 - self.config.min_prob_clamp)
            prior_logit = math.log(prior_clamped / (1.0 - prior_clamped))
            raw_logit = raw_logit - prior_logit
        return float(min(max(raw_logit, -self.config.logit_clamp), self.config.logit_clamp))

    def reconstruct(
        self,
        fragments: Sequence[ObservableFragment],
        candidate_edges: List[Tuple[int, int]],
        p_next_dict: Dict[Tuple[int, int], float],
        p_same_dict: Dict[Tuple[int, int], float],
    ) -> OptimizationOutput:
        """Execute full constrained reconstruction over observable fragments."""
        t_start = time.time()
        num_nodes = len(fragments)

        if num_nodes == 0:
            empty_res = MILPSolverResult(
                status="OPTIMAL",
                success=True,
                objective_value=0.0,
                optimality_gap=0.0,
                runtime_seconds=0.0,
                selected_edges=[],
                selected_gaps=[],
                selected_seeds=[],
                selected_tails=[],
                sequence_orders={},
                message="Zero fragments supplied",
            )
            return OptimizationOutput(
                reconstructed_files=[],
                solver_result=empty_res,
                num_nodes=0,
                num_candidate_edges=0,
                num_gap_edges=0,
                num_constraints=0,
                num_variables=0,
                runtime_seconds=0.0,
                format_evidence={},
            )

        # 1. Format Evidence & Boundary Extraction
        header_nodes: Set[int] = set()
        footer_nodes: Set[int] = set()
        evidence_map: Dict[int, FormatEvidence] = {}
        incompatible_pairs: Set[Tuple[int, int]] = set()

        if self.config.enable_format_constraints:
            header_nodes, footer_nodes, evidence_map = self.format_extractor.extract_boundaries(fragments)
            incompatible_pairs = self.format_extractor.find_incompatible_pairs(fragments, evidence_map)

        # 2. Candidate Gap Edges Generation
        gap_edges_info: Dict[Tuple[int, int], GapEdgeInfo] = {}
        all_candidate_edges = list(candidate_edges)

        if self.config.enable_gaps:
            for u, v in all_candidate_edges:
                p_s = p_same_dict.get((u, v), 0.5)
                p_n = p_next_dict.get((u, v), 0.5)
                logit_same = self._prob_to_logit(p_s)
                gap_info = self.gap_model.evaluate_gap_candidate(
                    u, v, fragments[u], fragments[v], p_s, p_n, logit_same
                )
                if gap_info is not None:
                    gap_edges_info[(u, v)] = gap_info

        # 3. Build Decision Variable Index Map
        vmap = OptimizationVariableBuilder.build_variable_map(
            num_nodes=num_nodes,
            candidate_edges=all_candidate_edges,
            enable_gaps=self.config.enable_gaps,
            enable_mtz_ordering=self.config.enable_mtz_ordering,
        )

        # 4. Construct Constraint Matrix
        c_builder = ConstraintMatrixBuilder(vmap)
        c_builder.build_flow_constraints()

        if self.config.enable_mtz_ordering:
            c_builder.build_mtz_subtour_constraints()

        if self.config.enable_format_constraints:
            c_builder.build_format_constraints(header_nodes, footer_nodes)
            c_builder.build_co_membership_guards(incompatible_pairs)

        # Incompatible threshold gating from P_same
        for u, v in all_candidate_edges:
            p_s = p_same_dict.get((u, v), 1.0)
            if p_s < self.config.tau_same_min:
                c_builder.build_co_membership_guards({(u, v)})

        A, lhs, rhs = c_builder.get_scipy_sparse_constraints()

        # 5. Assemble Objective Function Vector
        # maximize: sum w_ij * x_ij + sum w_gap_ij * g_ij + sum s_bonus * s_i + sum t_bonus * t_i - lambda * sum s_i
        obj_weights = np.zeros(vmap.total_variables, dtype=float)

        # 5a. Direct transitions: x_ij
        for (u, v), x_idx in vmap.edge_to_x_idx.items():
            p_n = p_next_dict.get((u, v), 0.5)
            obj_weights[x_idx] = self._prob_to_logit(p_n, prior=self.config.tau_prior)

        # 5b. Gap transitions: g_ij
        if self.config.enable_gaps:
            for (u, v), g_idx in vmap.edge_to_g_idx.items():
                if (u, v) in gap_edges_info:
                    obj_weights[g_idx] = gap_edges_info[(u, v)].net_weight
                else:
                    # Default gap penalty if not individually calculated
                    base_w = obj_weights[vmap.edge_to_x_idx[(u, v)]]
                    obj_weights[g_idx] = base_w - self.config.base_gap_penalty

        # 5c. Head (s_i) and Tail (t_i) bonuses & file penalty
        s_bonuses, t_bonuses = self.format_extractor.compute_seed_tail_bonuses(num_nodes, evidence_map)
        for i in range(num_nodes):
            s_idx = vmap.node_to_s_idx[i]
            t_idx = vmap.node_to_t_idx[i]
            # File regularizer penalty subtracts from s_i to discourage fragmenting files unnecessarily
            obj_weights[s_idx] = s_bonuses[i] - self.config.file_penalty
            obj_weights[t_idx] = t_bonuses[i]

        # 6. Execute Discrete Solver
        if self.config.solver_type == "cp_sat":
            solver = CPSATReconstructionSolver(time_limit_seconds=self.config.time_limit_seconds)
        else:
            solver = MILPReconstructionSolver(
                time_limit_seconds=self.config.time_limit_seconds,
                mip_rel_gap=self.config.mip_rel_gap,
            )

        solver_res = solver.solve(vmap, obj_weights, A, lhs, rhs)

        # 7. Trace Reconstructed Paths from Solver Solution
        reconstructed_files = self._extract_reconstructed_files(
            fragments=fragments,
            solver_res=solver_res,
            evidence_map=evidence_map,
            gap_edges_info=gap_edges_info,
        )

        total_runtime = time.time() - t_start

        diagnostics = {
            "solver_type": self.config.solver_type,
            "status": solver_res.status,
            "optimality_gap": solver_res.optimality_gap,
            "objective_value": solver_res.objective_value,
            "num_reconstructed_files": len(reconstructed_files),
            "num_selected_edges": len(solver_res.selected_edges),
            "num_selected_gaps": len(solver_res.selected_gaps),
            "num_headers_detected": len(header_nodes),
            "num_footers_detected": len(footer_nodes),
            "runtime_solver": solver_res.runtime_seconds,
            "runtime_total": total_runtime,
        }

        return OptimizationOutput(
            reconstructed_files=reconstructed_files,
            solver_result=solver_res,
            num_nodes=num_nodes,
            num_candidate_edges=len(all_candidate_edges),
            num_gap_edges=len(gap_edges_info),
            num_constraints=A.shape[0],
            num_variables=vmap.total_variables,
            runtime_seconds=total_runtime,
            format_evidence=evidence_map,
            diagnostics=diagnostics,
        )

    def _extract_reconstructed_files(
        self,
        fragments: Sequence[ObservableFragment],
        solver_res: MILPSolverResult,
        evidence_map: Dict[int, FormatEvidence],
        gap_edges_info: Dict[Tuple[int, int], GapEdgeInfo],
    ) -> List[ReconstructedFile]:
        """Trace path components from active edges and assemble ReconstructedFile instances."""
        if not solver_res.success:
            return []

        n = len(fragments)
        # Adjacency map: u -> (v, is_gap)
        succ: Dict[int, Tuple[int, bool]] = {}
        pred: Dict[int, Tuple[int, bool]] = {}

        for u, v in solver_res.selected_edges:
            succ[u] = (v, False)
            pred[v] = (u, False)

        for u, v in solver_res.selected_gaps:
            succ[u] = (v, True)
            pred[v] = (u, True)

        # Identify starting nodes (seeds s_i = 1 or nodes with in-degree 0 that have successors)
        start_nodes: Set[int] = set(solver_res.selected_seeds)
        for i in range(n):
            if i in succ and i not in pred:
                start_nodes.add(i)

        # If a node is completely isolated but was selected as both seed and tail
        for i in set(solver_res.selected_seeds).intersection(set(solver_res.selected_tails)):
            start_nodes.add(i)

        visited: Set[int] = set()
        reconstructed: List[ReconstructedFile] = []
        file_counter = 0

        # Sort start nodes for deterministic reconstruction ordering
        ordered_starts = sorted(list(start_nodes))

        for start in ordered_starts:
            if start in visited:
                continue

            curr = start
            seq_ids: List[str] = []
            byte_segments: List[bytes] = []
            claimed_format: Optional[str] = None

            # Check if start node has format evidence
            ev_start = evidence_map.get(curr)
            if ev_start and ev_start.detected_format:
                claimed_format = ev_start.detected_format

            while curr is not None and curr not in visited:
                visited.add(curr)
                frag = fragments[curr]
                seq_ids.append(frag.fragment_id)
                byte_segments.append(frag.fragment_bytes)

                # Update claimed format if discovered along the path
                if claimed_format is None:
                    ev_curr = evidence_map.get(curr)
                    if ev_curr and ev_curr.detected_format:
                        claimed_format = ev_curr.detected_format

                if curr in succ:
                    nxt, is_gap = succ[curr]
                    if is_gap:
                        # Insert gap placeholder
                        gap_info = gap_edges_info.get((curr, nxt))
                        gap_bytes_len = gap_info.estimated_missing_bytes if gap_info else 4096
                        seq_ids.append(f"[GAP_{fragments[curr].fragment_id}_{fragments[nxt].fragment_id}]")
                        byte_segments.append(MissingFragmentModel.create_gap_placeholder(gap_bytes_len))
                    curr = nxt
                else:
                    curr = None

            if seq_ids:
                recon_bytes = b"".join(byte_segments)
                recon_file = ReconstructedFile(
                    recon_id=f"recon_opt_{file_counter:03d}",
                    fragment_sequence=seq_ids,
                    reconstructed_bytes=recon_bytes,
                    claimed_type=claimed_format,
                )
                reconstructed.append(recon_file)
                file_counter += 1

        # Handle any leftover fragments that were selected in edges but missed due to potential cycle
        for i in range(n):
            if (i in succ or i in pred) and i not in visited:
                # Disjoint cycle recovery
                curr = i
                cycle_ids: List[str] = []
                cycle_bytes: List[bytes] = []
                while curr is not None and curr not in visited:
                    visited.add(curr)
                    frag = fragments[curr]
                    cycle_ids.append(frag.fragment_id)
                    cycle_bytes.append(frag.fragment_bytes)
                    if curr in succ:
                        nxt, _ = succ[curr]
                        curr = nxt
                    else:
                        curr = None
                if cycle_ids:
                    recon_file = ReconstructedFile(
                        recon_id=f"recon_opt_{file_counter:03d}",
                        fragment_sequence=cycle_ids,
                        reconstructed_bytes=b"".join(cycle_bytes),
                        claimed_type=None,
                    )
                    reconstructed.append(recon_file)
                    file_counter += 1

        return reconstructed
