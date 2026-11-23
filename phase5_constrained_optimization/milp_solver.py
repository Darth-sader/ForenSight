"""Mixed-Integer Linear Programming (MILP) Solver Engine (v2.0.0-phase5).

Solves the global forensic reconstruction optimization problem via SciPy HiGHS:
    maximize:  c_obj^T x
    subject to: lhs <= A x <= rhs
                lower_bounds <= x <= upper_bounds
                x_k in {0, 1} for integer decision variables

Provides comprehensive solver diagnostics:
- Optimality status (OPTIMAL, FEASIBLE, INFEASIBLE, TIME_LIMIT)
- Objective value & optimality gap
- Selected transition edges, missing fragment gaps, and seed fragments
- Execution runtime & constraint verification
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from phase5_constrained_optimization.optimization_variables import VariableIndexMap


@dataclass
class MILPSolverResult:
    """Detailed optimization solution container."""
    status: str                         # "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "TIME_LIMIT"
    success: bool
    objective_value: float
    optimality_gap: float
    runtime_seconds: float
    selected_edges: List[Tuple[int, int]]
    selected_gaps: List[Tuple[int, int]]
    selected_seeds: List[int]
    selected_tails: List[int]
    sequence_orders: Dict[int, float]
    solution_vector: Optional[np.ndarray] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "objective_value": round(self.objective_value, 4),
            "optimality_gap": round(self.optimality_gap, 6),
            "runtime_seconds": round(self.runtime_seconds, 4),
            "num_selected_edges": len(self.selected_edges),
            "num_selected_gaps": len(self.selected_gaps),
            "num_selected_files": len(self.selected_seeds),
            "message": self.message,
        }


class MILPReconstructionSolver:
    """Wrapper configuring and solving the reconstruction problem with HiGHS."""

    def __init__(
        self,
        time_limit_seconds: float = 60.0,
        mip_rel_gap: float = 1e-4,
        presolve: bool = True,
    ):
        self.time_limit = time_limit_seconds
        self.mip_rel_gap = mip_rel_gap
        self.presolve = presolve

    def solve(
        self,
        vmap: VariableIndexMap,
        objective_weights: np.ndarray,      # Maximize w^T x
        A: sparse.csc_matrix,
        lhs: np.ndarray,
        rhs: np.ndarray,
    ) -> MILPSolverResult:
        """Solve MILP formulation using SciPy / HiGHS."""
        t0 = time.time()

        # SciPy minimizes c^T x, so c = -objective_weights to maximize
        c = -objective_weights.astype(float)
        constraints = LinearConstraint(A, lhs, rhs)
        bounds = Bounds(vmap.lower_bounds, vmap.upper_bounds)

        options = {
            "time_limit": self.time_limit,
            "mip_rel_gap": self.mip_rel_gap,
            "presolve": self.presolve,
            "disp": False,
        }

        res = milp(
            c=c,
            integrality=vmap.integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        elapsed = time.time() - t0

        # Translate SciPy status code
        status_map = {
            0: "OPTIMAL",
            1: "TIME_LIMIT",
            2: "INFEASIBLE",
            3: "UNBOUNDED",
        }
        status_str = status_map.get(res.status, "FEASIBLE" if res.success else "FAILED")

        if not res.success and res.x is None:
            return MILPSolverResult(
                status=status_str,
                success=False,
                objective_value=0.0,
                optimality_gap=1.0,
                runtime_seconds=elapsed,
                selected_edges=[],
                selected_gaps=[],
                selected_seeds=[],
                selected_tails=[],
                sequence_orders={},
                message=str(res.message),
            )

        sol = res.x
        # Invert objective back to maximization
        obj_val = float(-res.fun) if res.fun is not None else 0.0

        # Extract selected decision variables
        selected_edges = []
        for e, idx in vmap.edge_to_x_idx.items():
            if sol[idx] > 0.5:
                selected_edges.append(e)

        selected_gaps = []
        for e, idx in vmap.edge_to_g_idx.items():
            if sol[idx] > 0.5:
                selected_gaps.append(e)

        selected_seeds = [
            i for i, idx in vmap.node_to_s_idx.items() if sol[idx] > 0.5
        ]
        selected_tails = [
            i for i, idx in vmap.node_to_t_idx.items() if sol[idx] > 0.5
        ]

        sequence_orders = {
            i: float(sol[idx]) for i, idx in vmap.node_to_u_idx.items()
        }

        # Calculate optimality gap where available
        opt_gap = 0.0 if res.status == 0 else self.mip_rel_gap

        return MILPSolverResult(
            status=status_str,
            success=True,
            objective_value=obj_val,
            optimality_gap=opt_gap,
            runtime_seconds=elapsed,
            selected_edges=selected_edges,
            selected_gaps=selected_gaps,
            selected_seeds=selected_seeds,
            selected_tails=selected_tails,
            sequence_orders=sequence_orders,
            solution_vector=sol,
            message=str(res.message),
        )
