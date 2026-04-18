"""
tests/test_grundy.py
====================
Unit tests for the grundy package.

Run with:
    pytest tests/
"""

import itertools
import networkx as nx
import pytest

from grundy.bb_grundy import (
    branch,
    branch2,
    branch3,
    counting_grundy_colorings,
    enumerate_orders,
    fast_stair_factor,
    greedy_coloring,
    is_greedy_coloring,
    lower_bound,
    lower_bound2,
    stair_factor,
    upper_bound,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def brute_force_grundy(G: nx.Graph) -> int:
    """Exact Grundy number via exhaustive permutation enumeration."""
    best = 0
    for perm in itertools.permutations(G.nodes()):
        col = greedy_coloring(G, perm)
        best = max(best, max(col.values()) + 1)
    return best


# ---------------------------------------------------------------------------
# Known-value fixtures
# ---------------------------------------------------------------------------

KNOWN = [
    ("P4",  nx.path_graph(4),                  3),
    ("P5",  nx.path_graph(5),                  3),
    ("P6",  nx.path_graph(6),                  3),
    ("C4",  nx.cycle_graph(4),                 2),
    ("C5",  nx.cycle_graph(5),                 3),
    ("C6",  nx.cycle_graph(6),                 3),
    ("K3",  nx.complete_graph(3),              3),
    ("K4",  nx.complete_graph(4),              4),
    ("K5",  nx.complete_graph(5),              5),
    ("K33", nx.complete_bipartite_graph(3, 3), 2),
]


# ---------------------------------------------------------------------------
# Greedy coloring
# ---------------------------------------------------------------------------

class TestGreedyColoring:
    def test_proper_coloring(self):
        G = nx.petersen_graph()
        order = list(G.nodes())
        col = greedy_coloring(G, order)
        for u, v in G.edges():
            assert col[u] != col[v], f"Edge ({u},{v}) monochromatic"

    def test_complete_graph(self):
        G = nx.complete_graph(5)
        col = greedy_coloring(G, list(G.nodes()))
        assert len(set(col.values())) == 5


# ---------------------------------------------------------------------------
# is_greedy_coloring
# ---------------------------------------------------------------------------

class TestIsGreedyColoring:
    def test_valid_grundy(self):
        G = nx.path_graph(4)      # 0-1-2-3
        # Order [1,0,2,3] → greedy coloring uses colors 0,1,0,1 but
        # order [0,2,1,3] gives a valid Grundy coloring with 3 colors.
        for perm in itertools.permutations(G.nodes()):
            col = greedy_coloring(G, perm)
            k = max(col.values()) + 1
            C = [[] for _ in range(k)]
            for v, c in col.items():
                C[c].append(v)
            assert is_greedy_coloring(G, C)

    def test_invalid_coloring(self):
        G = nx.path_graph(4)
        # Artificially put vertex 3 in class 2 without a neighbour in class 1.
        C = [[0, 2], [1], [3]]
        # 3 is not adjacent to 2 (only to 2? path is 0-1-2-3 so 3 adj to 2)
        # This should still be valid; let us make a clearly invalid one:
        C_bad = [[0, 3], [1], [2]]   # 3 in class 0 but 3-2 edge means
        # 3 should have neighbour in class 0 if it's in class 0 — ok.
        # Actually test: put vertex in higher class without lower-class neighbour.
        C_bad2 = [[0, 1, 2, 3]]  # trivial one-color: valid for χ=1 only if independent set
        # 0-1 is an edge so this is not a proper coloring → is_greedy should return False
        # because vertices sharing a class are adjacent.
        # is_greedy_coloring checks the greedy condition but not properness directly;
        # a proper-coloring check is implicit only when i=0 (no earlier classes needed).
        # The real invalid case: vertex in class i=1 with no neighbour in class 0.
        C_bad3 = [[0], [3]]   # 3 has no edge to 0 in P4 (edges: 0-1,1-2,2-3)
        assert not is_greedy_coloring(G, C_bad3)


# ---------------------------------------------------------------------------
# Upper bounds
# ---------------------------------------------------------------------------

class TestBounds:
    @pytest.mark.parametrize("name,G,gamma", KNOWN)
    def test_upper_bound_valid(self, name, G, gamma):
        assert upper_bound(G) >= gamma, f"{name}: upper_bound below Γ"

    @pytest.mark.parametrize("name,G,gamma", KNOWN)
    def test_stair_factor_valid(self, name, G, gamma):
        assert stair_factor(G) >= gamma, f"{name}: stair_factor below Γ"

    @pytest.mark.parametrize("name,G,gamma", KNOWN)
    def test_fast_stair_factor_matches(self, name, G, gamma):
        assert fast_stair_factor(G) == stair_factor(G), \
            f"{name}: fast_stair_factor != stair_factor"

    @pytest.mark.parametrize("name,G,gamma", KNOWN)
    def test_lower_bounds_valid(self, name, G, gamma):
        assert lower_bound(G)  <= gamma, f"{name}: lower_bound above Γ"
        assert lower_bound2(G) <= gamma, f"{name}: lower_bound2 above Γ"


# ---------------------------------------------------------------------------
# Branch-and-bound solvers
# ---------------------------------------------------------------------------

SOLVERS = [
    ("branch",  branch),
    ("branch2", branch2),
    ("branch3", branch3),
]


class TestBranchAndBound:
    @pytest.mark.parametrize("name,G,gamma", KNOWN)
    @pytest.mark.parametrize("solver_name,solver", SOLVERS)
    def test_known_gamma(self, name, G, gamma, solver_name, solver):
        result = solver(G)
        assert result["gamma"] == gamma, \
            f"{solver_name} on {name}: got {result['gamma']}, expected {gamma}"

    @pytest.mark.parametrize("solver_name,solver", SOLVERS)
    def test_valid_coloring(self, solver_name, solver):
        G = nx.erdos_renyi_graph(8, 0.4, seed=7)
        result = solver(G)
        assert result["valid"], f"{solver_name}: coloring is not a valid Grundy coloring"

    @pytest.mark.parametrize("solver_name,solver", SOLVERS)
    def test_agreement_with_brute_force(self, solver_name, solver):
        G = nx.erdos_renyi_graph(8, 0.3, seed=42)
        assert solver(G)["gamma"] == brute_force_grundy(G)


# ---------------------------------------------------------------------------
# enumerate_orders
# ---------------------------------------------------------------------------

class TestEnumerateOrders:
    @pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
    def test_matches_brute_force_count(self, seed):
        G = nx.erdos_renyi_graph(7, 0.2, seed=seed)
        bf  = counting_grundy_colorings(G)
        eo  = enumerate_orders(G)
        assert len(eo) == bf, \
            f"seed={seed}: enumerate_orders={len(eo)}, brute_force={bf}"

    def test_all_results_are_grundy(self):
        G = nx.path_graph(5)
        for partition in enumerate_orders(G):
            C = [list(cls) for cls in partition]
            assert is_greedy_coloring(G, C), \
                f"Non-Grundy partition returned: {partition}"

    def test_path_p4(self):
        G = nx.path_graph(4)
        bf = counting_grundy_colorings(G)
        eo = enumerate_orders(G)
        assert len(eo) == bf

    def test_complete_k4(self):
        G = nx.complete_graph(4)
        bf = counting_grundy_colorings(G)
        eo = enumerate_orders(G)
        assert len(eo) == bf

    def test_cycle_c5(self):
        G = nx.cycle_graph(5)
        bf = counting_grundy_colorings(G)
        eo = enumerate_orders(G)
        assert len(eo) == bf
