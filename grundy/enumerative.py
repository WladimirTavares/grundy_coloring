# ---------------------------------------------------------------------------
# Enumerative algorithms
# ---------------------------------------------------------------------------

import time
import itertools
import math
import heapq
import networkx as nx
from greedy_coloring import greedy_coloring, is_greedy_coloring

def enumerating_grundy_colorings(G: nx.Graph) -> dict:
    """Count all distinct Grundy colorings of *G* by brute force.

    Iterates over all n! vertex orderings, applies :func:`greedy_coloring` to
    each, verifies the result with :func:`is_greedy_coloring`, and collects
    distinct colorings (as frozensets of frozensets of color classes).

    This function is intended solely as a correctness reference for
    :func:`enumerate_orders`.  It is feasible only for n ≤ 10–12.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    int
        The number of distinct Grundy colorings of *G*.

    Complexity
    ----------
    O(n! · (n + m)).

    See Also
    --------
    enumerate_orders : More efficient enumerative algorithm.
    """
    start = time.time()
    nodes    = list(G.nodes())
    colorings: set = set()
    LB = 0
    bestC = []
    bb_nodes = 0
    
    for perm in itertools.permutations(nodes):
        bb_nodes += 1
        raw = greedy_coloring(G, list(perm))
        k   = max(raw.values()) + 1
        C   = [[] for _ in range(k)]
        for v, c in raw.items():
            C[c].append(v)
        if is_greedy_coloring(G, C):
            if len(C) > LB:
                LB = len(C)
                bestC = C
            key = tuple(tuple(sorted(cls)) for cls in C)
            colorings.add(key)
    
    return {
        "model":    "Permutation",
        "gamma":    LB,
        "cpu_s":    time.time() - start,
        "classes":  bestC,
        "valid":    is_greedy_coloring(G, bestC),
        "grundy_colorings": len(colorings),
    }


if __name__ == "__main__":

    G = nx.erdos_renyi_graph(8, 0.5, 1)

    print( enumerating_grundy_colorings(G) )