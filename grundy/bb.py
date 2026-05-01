
"""
Branch-and-bound solvers for computing the Grundy
(first-fit chromatic) number of an undirected graph, together with an
enumerative algorithm that counts all distinct Grundy colorings without
exhaustive permutation search.

Background
----------
Given an undirected graph G = (V, E), a **greedy coloring** produced by a
vertex ordering σ = (v₁, v₂, …, vₙ) assigns to each vᵢ the smallest
non-negative integer color not used by any earlier neighbour.  The
**Grundy number** Γ(G) is the maximum number of colors over all orderings:

    Γ(G) = max_{σ ∈ Sₙ}  χ_greedy(G, σ)

It satisfies the classical chain:

    χ(G) ≤ Γ(G) ≤ Δ(G) + 1

A coloring C = (C₀, C₁, …, C_{k-1}) is a **Grundy coloring** if and only
if every vertex v ∈ Cᵢ has at least one neighbour in each earlier class Cⱼ
(j < i).  This property underlies all algorithms in this module.

Algorithms
----------
Branch-and-bound (exact, combinatorial):
*    branch_and_bound(G, ub_func)  – B&B with in-place graph mutation
"""


import time
import itertools
from collections import defaultdict, deque
from typing import Callable, Optional
import math
import heapq
import networkx as nx
from upper_bound import fast_stair_factor
from lower_bound import lb_reverse_sl, lb_reverse_lf
from greedy_coloring import is_greedy_coloring



def find_cliques(G: nx.Graph, nodes: Optional[list] = None):
    """Enumerate all maximal cliques of the undirected graph *G*.

    Uses the Bron–Kerbosch algorithm with Tomita–Tanaka–Takahashi pivot
    selection (maximising |P ∩ N(u)|), implemented *iteratively* to avoid
    Python recursion-depth limits on large inputs.

    In the branch-and-bound solvers, ``find_cliques`` is called on the
    **complement** graph G̅ to enumerate all maximal independent sets of G,
    since a maximal clique of G̅ is exactly a maximal independent set of G.

    Parameters
    ----------
    G : nx.Graph
        An undirected graph.  Self-loops are ignored.
    nodes : list, optional
        If provided, only yield maximal cliques that contain all vertices in
        this list.  The list must itself form a clique in *G*; a
        ``ValueError`` is raised otherwise.

    Yields
    ------
    list
        Each yielded list is a maximal clique of *G* (or a maximal extension
        of the seed *nodes*).

    Complexity
    ----------
    O(3^{n/3}) in the worst case (Moon–Moser bound [MM65]).

    Notes
    -----
    The pivot is chosen to maximise the number of candidates eliminated at
    each step, following the analysis of Tomita et al. [TTT06] and the
    correction of Cazals & Karande [CK08].

    References
    ----------
    .. [BK73]  Bron & Kerbosch (1973).
    .. [TTT06] Tomita, Tanaka & Takahashi (2006).
    .. [CK08]  Cazals & Karande (2008).
    .. [MM65]  Moon & Moser (1965).
    """
    if len(G) == 0:
        return

    adj = {u: {v for v in G[u] if v != u} for u in G}
    Q = nodes[:] if nodes is not None else []
    cand = set(G)
    for node in Q:
        if node not in cand:
            raise ValueError(f"The given `nodes` {nodes} do not form a clique")
        cand &= adj[node]

    if not cand:
        yield Q[:]
        return

    subg = cand.copy()
    stack = []
    Q.append(None)
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass


# ---------------------------------------------------------------------------
# Branch-and-bound solvers  (generic + three specialised variants)
# ---------------------------------------------------------------------------

from upper_bound import delta_1, delta_2, fast_stair_factor

def branch_and_bound(
    G: nx.Graph,
    time_limit = math.inf
) -> dict:
    """Compute Γ(G) via branch-and-bound with in-place graph mutation.

    This is the **recommended variant** for general use.  Instead of copying
    the subgraph at each recursive call it removes the chosen color class from
    G in-place and restores it (backtrack) after the recursive call returns.
    This avoids all allocation overhead at the cost of a small bookkeeping step
    to save and restore the removed edges.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.  **The graph is modified in place during
        the search but is fully restored before the function returns.**
    ub_func : callable, optional
        Upper-bound function; defaults to :func:`fast_stair_factor`.

    Returns
    -------
    dict
        Same keys as :func:`branch_and_bound`.

    Notes
    -----
    Because the graph is mutated in place, this function is **not
    thread-safe**.  Run it in a single-threaded context or pass independent
    copies to each thread.

    See Also
    --------
    branch_and_bound  : Graph-copy variant (thread-safe, higher memory use).
    branch_and_bound2 : Frozenset-view variant.
    branch3           : Convenience wrapper for this function.

    References
    ----------
    .. [BFK18] Bonnet et al. (2018), §3.
    """
    lb1 = lb_reverse_sl(G)
    lb2 = lb_reverse_lf(G)
    LB    = max(lb1["lower_bound"], lb2["lower_bound"])
    bestC = lb1["coloring"] if lb1["lower_bound"] >= lb2["lower_bound"] else lb2["coloring"]
    bb_nodes = 0
    pruned = 0
    deadline = time.time() + time_limit

    def expand(H: nx.Graph, C: list[list]) -> None:
        nonlocal LB, bb_nodes, bestC, pruned
        
        if time.time () > deadline:
            return 

        if len(H) == 0:
            if len(C) > LB:
                LB    = len(C)
                bestC = C
            return
        
        if len(C) + len(H.nodes()) <= LB:
            pruned += 1
            return 
        
        if len(C) + delta_1(G) <= LB:
            pruned += 1
            return 

        if len(C) + delta_2(H) <= LB:
            pruned += 1
            return
        
        if len(C) + fast_stair_factor(H) <= LB:
            pruned += 1
            return
        

        bb_nodes += 1
        Hc = nx.complement(H)
        for clique in find_cliques(Hc):
            removed_edges = [(u, v) for u in clique for v in list(H.neighbors(u))]
            H.remove_nodes_from(clique)
            expand(H, C + [clique])
            H.add_nodes_from(clique)
            H.add_edges_from(removed_edges)

    start = time.time()
    expand(G, [])
    return {
        "model":    "BB/in-place",
        "gamma":    LB,
        "cpu_s":    time.time() - start,
        "classes":  bestC,
        "valid":    is_greedy_coloring(G, bestC),
        "bb_nodes": bb_nodes,
        "pruned":       pruned,

    }


if __name__ == "__main__":

    G = nx.erdos_renyi_graph(25, 0.5, 1)

    print( branch_and_bound(G, 5) )