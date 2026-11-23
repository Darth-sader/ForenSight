"""Optimization Constraints Matrix Generator (v2.0.0-phase5).

Builds linear constraints in sparse / standard matrix representation:
1. Out-degree constraint: sum_j (x_ij + g_ij) + t_i <= 1
2. In-degree constraint: sum_j (x_ji + g_ji) + s_i <= 1
3. Flow conservation: in_flow(i) + s_i == out_flow(i) + t_i
4. Miller-Tucker-Zemlin (MTZ) Subtour / Cycle Elimination:
       u_i - u_j + n * (x_ij + g_ij) <= n - 1
5. Seed sequence boundary: u_i <= n * (1 - s_i)
6. Format boundary guards (header in-degree = 0, footer out-degree = 0)
7. Co-membership threshold gating: x_ij == 0 when P_same < tau_min
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

from phase5_constrained_optimization.optimization_variables import VariableIndexMap


class ConstraintMatrixBuilder:
    """Constructs constraint matrices and bounds vectors for SciPy / HiGHS MILP."""

    def __init__(self, vmap: VariableIndexMap):
        self.vmap = vmap
        self.rows: List[Dict[int, float]] = []
        self.lower_bounds: List[float] = []
        self.upper_bounds: List[float] = []

    def add_constraint(
        self,
        row_dict: Dict[int, float],
        lb: float = -np.inf,
        ub: float = np.inf,
    ) -> None:
        """Add a single linear constraint: lb <= sum_k a_k * x_k <= ub."""
        if not row_dict:
            return
        self.rows.append(row_dict)
        self.lower_bounds.append(lb)
        self.upper_bounds.append(ub)

    def build_flow_constraints(self) -> None:
        """Add in-degree, out-degree, and path flow conservation constraints."""
        n = self.vmap.num_nodes
        edges = self.vmap.candidate_edges

        # Group edges by source and target
        out_edges: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n)}
        in_edges: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n)}

        for u, v in edges:
            out_edges[u].append((u, v))
            in_edges[v].append((u, v))

        for i in range(n):
            # 1. Out-degree bound: sum_j (x_ij + g_ij) + t_i <= 1
            out_dict = {}
            for e in out_edges[i]:
                out_dict[self.vmap.edge_to_x_idx[e]] = 1.0
                if e in self.vmap.edge_to_g_idx:
                    out_dict[self.vmap.edge_to_g_idx[e]] = 1.0
            out_dict[self.vmap.node_to_t_idx[i]] = 1.0
            self.add_constraint(out_dict, lb=0.0, ub=1.0)

            # 2. In-degree bound: sum_j (x_ji + g_ji) + s_i <= 1
            in_dict = {}
            for e in in_edges[i]:
                in_dict[self.vmap.edge_to_x_idx[e]] = 1.0
                if e in self.vmap.edge_to_g_idx:
                    in_dict[self.vmap.edge_to_g_idx[e]] = 1.0
            in_dict[self.vmap.node_to_s_idx[i]] = 1.0
            self.add_constraint(in_dict, lb=0.0, ub=1.0)

            # 3. Flow conservation: in_flow(i) + s_i - out_flow(i) - t_i == 0
            flow_dict = {}
            for e in in_edges[i]:
                flow_dict[self.vmap.edge_to_x_idx[e]] = 1.0
                if e in self.vmap.edge_to_g_idx:
                    flow_dict[self.vmap.edge_to_g_idx[e]] = 1.0
            flow_dict[self.vmap.node_to_s_idx[i]] = 1.0

            for e in out_edges[i]:
                flow_dict[self.vmap.edge_to_x_idx[e]] = flow_dict.get(self.vmap.edge_to_x_idx[e], 0.0) - 1.0
                if e in self.vmap.edge_to_g_idx:
                    flow_dict[self.vmap.edge_to_g_idx[e]] = flow_dict.get(self.vmap.edge_to_g_idx[e], 0.0) - 1.0
            flow_dict[self.vmap.node_to_t_idx[i]] = flow_dict.get(self.vmap.node_to_t_idx[i], 0.0) - 1.0

            self.add_constraint(flow_dict, lb=0.0, ub=0.0)

    def build_mtz_subtour_constraints(self) -> None:
        """Add Miller-Tucker-Zemlin subtour elimination constraints: u_i - u_j + n * x_ij <= n - 1."""
        if not self.vmap.node_to_u_idx:
            return

        n = self.vmap.num_nodes
        edges = self.vmap.candidate_edges

        for u, v in edges:
            if u == v:
                continue
            u_idx = self.vmap.node_to_u_idx[u]
            v_idx = self.vmap.node_to_u_idx[v]
            x_idx = self.vmap.edge_to_x_idx[(u, v)]

            # u_i - u_j + n * x_ij <= n - 1
            row_dict = {
                u_idx: 1.0,
                v_idx: -1.0,
                x_idx: float(n),
            }
            if (u, v) in self.vmap.edge_to_g_idx:
                row_dict[self.vmap.edge_to_g_idx[(u, v)]] = float(n)

            self.add_constraint(row_dict, lb=-np.inf, ub=float(n - 1))

        # Anchor seed ordering: when s_i = 1, u_i <= 0
        for i in range(n):
            u_idx = self.vmap.node_to_u_idx[i]
            s_idx = self.vmap.node_to_s_idx[i]
            # u_i - n * (1 - s_i) <= 0  <=>  u_i + n * s_i <= n
            self.add_constraint({u_idx: 1.0, s_idx: float(n)}, lb=-np.inf, ub=float(n))

    def build_format_constraints(
        self,
        header_nodes: Set[int],
        footer_nodes: Set[int],
    ) -> None:
        """Hard format constraints: headers cannot have in-edges; footers cannot have out-edges."""
        edges = self.vmap.candidate_edges

        for u, v in edges:
            # If v is a known file header, no edge can enter v
            if v in header_nodes:
                self.add_constraint({self.vmap.edge_to_x_idx[(u, v)]: 1.0}, lb=0.0, ub=0.0)
                if (u, v) in self.vmap.edge_to_g_idx:
                    self.add_constraint({self.vmap.edge_to_g_idx[(u, v)]: 1.0}, lb=0.0, ub=0.0)

            # If u is a known file footer, no edge can exit u
            if u in footer_nodes:
                self.add_constraint({self.vmap.edge_to_x_idx[(u, v)]: 1.0}, lb=0.0, ub=0.0)
                if (u, v) in self.vmap.edge_to_g_idx:
                    self.add_constraint({self.vmap.edge_to_g_idx[(u, v)]: 1.0}, lb=0.0, ub=0.0)

    def build_co_membership_guards(
        self,
        incompatible_pairs: Set[Tuple[int, int]],
    ) -> None:
        """Hard exclusion: zero out transitions between fragments from provably different files."""
        for u, v in incompatible_pairs:
            if (u, v) in self.vmap.edge_to_x_idx:
                self.add_constraint({self.vmap.edge_to_x_idx[(u, v)]: 1.0}, lb=0.0, ub=0.0)
            if (u, v) in self.vmap.edge_to_g_idx:
                self.add_constraint({self.vmap.edge_to_g_idx[(u, v)]: 1.0}, lb=0.0, ub=0.0)

    def get_scipy_sparse_constraints(self) -> Tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
        """Convert rows into SciPy csc_matrix, lhs (lb), and rhs (ub)."""
        num_constraints = len(self.rows)
        num_vars = self.vmap.total_variables

        if num_constraints == 0:
            return sparse.csc_matrix((0, num_vars)), np.zeros(0), np.zeros(0)

        row_indices = []
        col_indices = []
        data_values = []

        for r_idx, row_dict in enumerate(self.rows):
            for c_idx, val in row_dict.items():
                row_indices.append(r_idx)
                col_indices.append(c_idx)
                data_values.append(val)

        A = sparse.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(num_constraints, num_vars),
            dtype=float,
        ).tocsc()

        return A, np.array(self.lower_bounds, dtype=float), np.array(self.upper_bounds, dtype=float)
