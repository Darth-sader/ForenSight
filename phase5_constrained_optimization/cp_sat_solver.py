"""Constraint Programming (CP-SAT) Solver Interface (v2.0.0-phase5).

Translates the forensic reconstruction problem into integer domain constraints:
1. Integer-scaled objective coefficients: c_int = int(round(c_float * 10000))
2. Boolean variables for edges, gaps, heads, and tails.
3. Linear constraints for flow conservation and MTZ subtours.
4. Solves with Google OR-Tools CP-SAT when installed, or gracefully falls back to HiGHS MILP.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from phase5_constrained_optimization.milp_solver import MILPReconstructionSolver, MILPSolverResult
from phase5_constrained_optimization.optimization_variables import VariableIndexMap


class CPSATReconstructionSolver:
    """CP-SAT Constraint Programming solver for multi-file reconstruction."""

    def __init__(
        self,
        time_limit_seconds: float = 60.0,
        scale_factor: int = 10000,
    ):
        self.time_limit = time_limit_seconds
        self.scale_factor = scale_factor
        self._has_ortools = False
        try:
            import ortools.sat.python.cp_model as cp_model
            self._has_ortools = True
        except ImportError:
            self._has_ortools = False

    def solve(
        self,
        vmap: VariableIndexMap,
        objective_weights: np.ndarray,
        A: Any,
        lhs: np.ndarray,
        rhs: np.ndarray,
    ) -> MILPSolverResult:
        """Solve using OR-Tools CP-SAT if available, else route to HiGHS MILP."""
        t0 = time.time()

        if not self._has_ortools:
            # Fallback to HiGHS MILP solver
            milp_solver = MILPReconstructionSolver(time_limit_seconds=self.time_limit)
            res = milp_solver.solve(vmap, objective_weights, A, lhs, rhs)
            res.message = f"Solved via HiGHS MILP (OR-Tools CP-SAT not installed in Python environment: {res.message})"
            return res

        # If OR-Tools is installed, formulate CP-SAT model
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        n_vars = vmap.total_variables
        cp_vars = []

        # Create CP variables
        for i in range(n_vars):
            is_int = vmap.integrality[i]
            lb = int(np.floor(vmap.lower_bounds[i]))
            ub = int(np.ceil(vmap.upper_bounds[i]))
            if is_int:
                cp_vars.append(model.NewIntVar(lb, ub, f"x_{i}"))
            else:
                cp_vars.append(model.NewIntVar(lb, ub, f"u_{i}"))

        # Add constraints from sparse matrix A
        A_csr = A.tocsr()
        for r in range(A_csr.shape[0]):
            row_start = A_csr.indptr[r]
            row_end = A_csr.indptr[r + 1]
            cols = A_csr.indices[row_start:row_end]
            vals = A_csr.data[row_start:row_end]

            expr = sum(int(round(vals[k])) * cp_vars[cols[k]] for k in range(len(cols)))
            lb = lhs[r]
            ub = rhs[r]
            if np.isneginf(lb) and np.isposinf(ub):
                continue
            elif np.isneginf(lb):
                model.Add(expr <= int(round(ub)))
            elif np.isposinf(ub):
                model.Add(expr >= int(round(lb)))
            elif abs(lb - ub) < 1e-6:
                model.Add(expr == int(round(lb)))
            else:
                model.Add(expr >= int(round(lb)))
                model.Add(expr <= int(round(ub)))

        # Scaled integer objective (maximize)
        obj_expr = sum(
            int(round(objective_weights[i] * self.scale_factor)) * cp_vars[i]
            for i in range(n_vars)
        )
        model.Maximize(obj_expr)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver_status = solver.Solve(model)
        elapsed = time.time() - t0

        status_str_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "INVALID",
            cp_model.UNKNOWN: "TIME_LIMIT",
        }
        status_name = status_str_map.get(solver_status, "UNKNOWN")
        success = solver_status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        if not success:
            return MILPSolverResult(
                status=status_name,
                success=False,
                objective_value=0.0,
                optimality_gap=1.0,
                runtime_seconds=elapsed,
                selected_edges=[],
                selected_gaps=[],
                selected_seeds=[],
                selected_tails=[],
                sequence_orders={},
                message="CP-SAT could not find a feasible solution within time limit",
            )

        sol_vals = np.array([solver.Value(cp_vars[i]) for i in range(n_vars)])
        obj_val = float(solver.ObjectiveValue() / self.scale_factor)

        selected_edges = [e for e, idx in vmap.edge_to_x_idx.items() if sol_vals[idx] > 0]
        selected_gaps = [e for e, idx in vmap.edge_to_g_idx.items() if sol_vals[idx] > 0]
        selected_seeds = [i for i, idx in vmap.node_to_s_idx.items() if sol_vals[idx] > 0]
        selected_tails = [i for i, idx in vmap.node_to_t_idx.items() if sol_vals[idx] > 0]
        seq_orders = {i: float(sol_vals[idx]) for i, idx in vmap.node_to_u_idx.items()}

        return MILPSolverResult(
            status=status_name,
            success=True,
            objective_value=obj_val,
            optimality_gap=0.0 if status_name == "OPTIMAL" else 0.01,
            runtime_seconds=elapsed,
            selected_edges=selected_edges,
            selected_gaps=selected_gaps,
            selected_seeds=selected_seeds,
            selected_tails=selected_tails,
            sequence_orders=seq_orders,
            solution_vector=sol_vals,
            message="CP-SAT solved successfully",
        )
