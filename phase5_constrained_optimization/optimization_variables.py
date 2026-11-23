"""Optimization Decision Variables and Index Mapping (v2.0.0-phase5).

Defines the mathematical decision variable index layout for:
1. Directed Adjacency Variables: x_ij in {0, 1}
2. Missing Fragment / Gap Variables: g_ij in {0, 1}
3. File Head / Start Variables: s_i in {0, 1}
4. File Tail / Termination Variables: t_i in {0, 1}
5. Topological Sequence Index Variables: u_i in [0, n] (MTZ subtour elimination)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class VariableIndexMap:
    """Manages bijective mapping between forensic entities and solver variable indices."""
    num_nodes: int
    candidate_edges: List[Tuple[int, int]]  # List of (u, v) pairs
    edge_to_x_idx: Dict[Tuple[int, int], int] = field(default_factory=dict)
    edge_to_g_idx: Dict[Tuple[int, int], int] = field(default_factory=dict)
    node_to_s_idx: Dict[int, int] = field(default_factory=dict)
    node_to_t_idx: Dict[int, int] = field(default_factory=dict)
    node_to_u_idx: Dict[int, int] = field(default_factory=dict)
    total_variables: int = 0
    integrality: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    lower_bounds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    upper_bounds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))


class OptimizationVariableBuilder:
    """Constructs the variable vector layout and bounds for the reconstruction MILP."""

    @staticmethod
    def build_variable_map(
        num_nodes: int,
        candidate_edges: List[Tuple[int, int]],
        enable_gaps: bool = True,
        enable_mtz_ordering: bool = True,
    ) -> VariableIndexMap:
        """Construct the flattened variable mapping."""
        vmap = VariableIndexMap(num_nodes=num_nodes, candidate_edges=candidate_edges)
        idx = 0

        # 1. Adjacency variables: x_ij in {0, 1}
        for u, v in candidate_edges:
            vmap.edge_to_x_idx[(u, v)] = idx
            idx += 1

        # 2. Gap variables: g_ij in {0, 1}
        if enable_gaps:
            for u, v in candidate_edges:
                vmap.edge_to_g_idx[(u, v)] = idx
                idx += 1

        # 3. Head / Start variables: s_i in {0, 1}
        for i in range(num_nodes):
            vmap.node_to_s_idx[i] = idx
            idx += 1

        # 4. Tail / Terminator variables: t_i in {0, 1}
        for i in range(num_nodes):
            vmap.node_to_t_idx[i] = idx
            idx += 1

        # 5. MTZ Topological sequence order variables: u_i in [0, num_nodes]
        if enable_mtz_ordering:
            for i in range(num_nodes):
                vmap.node_to_u_idx[i] = idx
                idx += 1

        vmap.total_variables = idx

        # Integrality: 1 for integer/binary, 0 for continuous
        integrality = np.ones(idx, dtype=int)
        lower_bounds = np.zeros(idx, dtype=float)
        upper_bounds = np.ones(idx, dtype=float)

        # MTZ variables u_i are continuous or integer in [0, num_nodes]
        if enable_mtz_ordering:
            for i in range(num_nodes):
                u_idx = vmap.node_to_u_idx[i]
                lower_bounds[u_idx] = 0.0
                upper_bounds[u_idx] = float(num_nodes)
                integrality[u_idx] = 0  # Can be relaxed to continuous in MTZ without losing integrality

        vmap.integrality = integrality
        vmap.lower_bounds = lower_bounds
        vmap.upper_bounds = upper_bounds

        return vmap
