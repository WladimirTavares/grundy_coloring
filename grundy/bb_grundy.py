"""
bb_grundy.py
============
Branch-and-bound algorithm for computing the Grundy (first-fit chromatic) number
of an undirected graph, together with an enumerative algorithm for counting all
distinct Grundy colorings.

The Grundy number Γ(G) of a graph G is the maximum number of colors used by any
greedy (first-fit) coloring of G, over all possible vertex orderings. It satisfies:

    χ(G) ≤ Γ(G) ≤ Δ(G) + 1

where χ(G) is the chromatic number and Δ(G) is the maximum degree.

The branch-and-bound approach exploits the recurrence:

    Γ(S) = max{ Γ(S \\ X) + 1  |  X ⊆ S is an independent dominating set of G[S] }

Three variants of the branch-and-bound solver are provided, differing only in the
upper-bound function used to prune the search tree:

    * ``branch``  – trivial upper bound  Δ(G[S]) + 1
    * ``branch2`` – stair-factor bound   (quadratic residue sequence)
    * ``branch3`` – fast stair-factor    (bucket-queue residue sequence, O(n + m))

An enumerative algorithm ``enumerate_orders`` is also provided. It generates a
superset of all proper colorings that are candidates for being Grundy colorings,
verifies each candidate, and collects the distinct ones.

References
----------
.. [BGK05] Shi, Z., Goddard, W., Hedetniemi, S. T., Kennedy, K., Laskar, R., &
           McRae, A. (2005). An algorithm for partial Grundy number on trees.
           *Discrete Mathematics*, 304(1-3), 108-116.
.. [BFK18] Bonnet, É., Foucaud, F., Kim, E. J., & Sikora, F. (2018).
           Complexity of Grundy coloring and its variants.
           *Discrete Applied Mathematics*, 243, 99-114.
.. [BK73]  Bron, C. and Kerbosch, J. (1973). Algorithm 457: finding all cliques
           of an undirected graph. *Communications of the ACM*, 16(9), 575-577.
.. [TTT06] Tomita, E., Tanaka, A., & Takahashi, H. (2006). The worst-case time
           complexity for generating all maximal cliques and computational
           experiments. *Theoretical Computer Science*, 363(1), 28-42.
.. [CK08]  Cazals, F., & Karande, C. (2008). A note on the problem of reporting
           maximal cliques. *Theoretical Computer Science*, 407(1-3), 564-568.
.. [MM65]  Moon, J. and Moser, L. (1965). On cliques in graphs.
           *Israel Journal of Mathematics*, 3(1), 23-28.
"""

import time
import itertools
from collections import defaultdict, deque
from typing import Optional

import networkx as nx


# ---------------------------------------------------------------------------
# Upper-bound functions
# ---------------------------------------------------------------------------

def upper_bound(G: nx.Graph) -> int:
    """Return the trivial upper bound Δ(G) + 1 on the Grundy number of *G*.

    The bound follows directly from the fact that, in any greedy coloring,
    every color class *i* (1-indexed) must contain at least one vertex of
    degree ≥ i − 1.  Therefore Γ(G) ≤ Δ(G) + 1, where Δ(G) is the
    maximum degree of *G*.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph (loops and parallel edges are ignored).

    Returns
    -------
    int
        The value Δ(G) + 1.

    Complexity
    ----------
    O(n), where n = |V(G)|.

    References
    ----------
    .. [BGK05] Shi et al. (2005), §2.
    """
    max_degree = max(dict(G.degree()).values())
    return max_degree + 1


def stair_factor(G: nx.Graph) -> int:
    """Return the *stair-factor* upper bound on the Grundy number of *G*.

    The stair factor is computed from the *degree sequence of the degeneracy
    ordering* (also called the *residue sequence*).  Concretely:

    1. Repeatedly remove the vertex of **maximum** residual degree and record
       that degree as d_i (i = 1, 2, …, n).
    2. The stair factor is  min_{i} (d_i + i).

    This bound is at least as tight as the trivial Δ(G) + 1 bound and is
    often significantly smaller.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    int
        The stair-factor upper bound k such that Γ(G) ≤ k.

    Complexity
    ----------
    O(n²) due to the ``defaultdict``-based degree tracking.  For an O(n + m)
    implementation see :func:`fast_stair_factor`.

    References
    ----------
    .. [BGK05] Shi et al. (2005), Theorem 3.
    """
    H = G.copy()
    degrees: defaultdict[int, set] = defaultdict(set)
    deg = dict(H.degree())

    max_deg = 0
    for v, d in deg.items():
        degrees[d].add(v)
        max_deg = max(max_deg, d)

    residue: list[int] = []

    def find_max_degree() -> int:
        for d in range(max_deg, -1, -1):
            if d in degrees and degrees[d]:
                return d
        return 0

    for _ in range(len(G)):
        d = find_max_degree()
        u = degrees[d].pop()
        if not degrees[d]:
            del degrees[d]
        residue.append(d)
        for v in list(H.neighbors(u)):
            dv = deg[v]
            degrees[dv].remove(v)
            if not degrees[dv]:
                del degrees[dv]
            deg[v] -= 1
            degrees[deg[v]].add(v)
        H.remove_node(u)
        del deg[u]

    k = float('inf')
    for i, d in enumerate(residue):
        k = min(k, d + i + 1)
    return int(k)


def fast_stair_factor(G: nx.Graph) -> int:
    """Return the stair-factor upper bound using a bucket-queue for O(n + m) time.

    Computes the same bound as :func:`stair_factor` but replaces the
    ``defaultdict``-based degree tracking with a flat bucket array (list of
    sets) indexed by degree value, achieving true O(n + m) time.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph with n vertices and m edges.

    Returns
    -------
    int
        The stair-factor upper bound k such that Γ(G) ≤ k.

    Complexity
    ----------
    O(n + m).

    References
    ----------
    .. [BGK05] Shi et al. (2005), Theorem 3.
    """
    H = G.copy()
    n = len(G)
    deg = dict(H.degree())
    buckets: list[set] = [set() for _ in range(n)]
    max_degree = 0
    for v, d in deg.items():
        buckets[d].add(v)
        if d > max_degree:
            max_degree = d

    residue: list[int] = []
    for _ in range(n):
        while max_degree >= 0 and not buckets[max_degree]:
            max_degree -= 1
        d = max_degree
        u = buckets[d].pop()
        residue.append(d)
        for v in list(H.neighbors(u)):
            dv = deg[v]
            buckets[dv].remove(v)
            deg[v] -= 1
            buckets[dv - 1].add(v)
        H.remove_node(u)
        del deg[u]

    k = float('inf')
    for i, d in enumerate(residue):
        k = min(k, d + i + 1)
    return int(k)


# ---------------------------------------------------------------------------
# Vertex ordering strategies
# ---------------------------------------------------------------------------

def strategy_largest_first(G: nx.Graph) -> list:
    """Return vertices of *G* ordered by decreasing residual degree (largest-first).

    At each step the vertex with the highest current residual degree is
    selected and removed, and the residual degrees of its neighbours are
    decremented.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    list
        Vertices ordered so that the first element had the highest residual
        degree at the time of its removal.

    Complexity
    ----------
    O(n + m) using a bucket-queue of size Δ(G).
    """
    H = G.copy()
    n = len(G)
    deg = dict(H.degree())
    buckets: list[set] = [set() for _ in range(n)]
    max_degree = 0
    for v, d in deg.items():
        buckets[d].add(v)
        if d > max_degree:
            max_degree = d

    result: list = []
    for _ in range(n):
        while max_degree >= 0 and not buckets[max_degree]:
            max_degree -= 1
        d = max_degree
        u = buckets[d].pop()
        result.append(u)
        for v in list(H.neighbors(u)):
            dv = deg[v]
            buckets[dv].remove(v)
            deg[v] -= 1
            buckets[dv - 1].add(v)
        H.remove_node(u)
        del deg[u]

    return result


def strategy_smallest_last(G: nx.Graph) -> list:
    """Return vertices of *G* ordered so that the vertex of minimum residual
    degree is placed **last** (smallest-last ordering).

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph with n vertices and m edges.

    Returns
    -------
    list
        Vertices such that the last element is the first vertex of minimum
        residual degree removed during the process.

    Complexity
    ----------
    O(n + m) using a bucket-queue with a maintained lower-bound pointer.
    """
    H = G.copy()
    result: deque = deque()
    degrees: defaultdict[int, set] = defaultdict(set)
    lbound = float("inf")
    for node, d in H.degree():
        degrees[d].add(node)
        lbound = min(lbound, d)

    def find_min_degree() -> int:
        return next(d for d in itertools.count(lbound) if d in degrees)

    for _ in G:
        min_degree = find_min_degree()
        u = degrees[min_degree].pop()
        if not degrees[min_degree]:
            del degrees[min_degree]
        result.appendleft(u)
        for v in H[u]:
            degree = H.degree(v)
            degrees[degree].remove(v)
            if not degrees[degree]:
                del degrees[degree]
            degrees[degree - 1].add(v)
        H.remove_node(u)
        lbound = min_degree - 1

    return list(result)


# ---------------------------------------------------------------------------
# Greedy coloring
# ---------------------------------------------------------------------------

def greedy_coloring(G: nx.Graph, nodes: list) -> dict:
    """Assign colors to vertices of *G* greedily in the given vertex order.

    Each vertex is assigned the smallest non-negative integer color not
    already used by any of its already-colored neighbours (first-fit rule).

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.
    nodes : list
        An ordering of all vertices of *G*.

    Returns
    -------
    dict
        A mapping ``{vertex: color}`` where colors are non-negative integers.

    Complexity
    ----------
    O(n + m).
    """
    colors: dict = {}
    for u in nodes:
        nbr_colors = {colors[v] for v in G[u] if v in colors}
        color = 0
        while color in nbr_colors:
            color += 1
        colors[u] = color
    return colors


# ---------------------------------------------------------------------------
# Greedy-coloring validator
# ---------------------------------------------------------------------------

def is_greedy_coloring(G: nx.Graph, C: list[list]) -> bool:
    """Check whether the partition *C* is a valid Grundy (greedy) coloring of *G*.

    A coloring C = (C_0, C_1, …, C_{k-1}) is *greedy* if and only if every
    vertex v ∈ C_i has at least one neighbour in each earlier color class C_j
    for j < i.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.
    C : list of list
        A partition of V(G) into color classes.

    Returns
    -------
    bool
        ``True`` if every vertex satisfies the greedy condition.

    Complexity
    ----------
    O(k · n · m) in the worst case.
    """
    for i in range(len(C)):
        for v in C[i]:
            for j in range(i):
                if not (set(G[v]) & set(C[j])):
                    return False
    return True


# ---------------------------------------------------------------------------
# Lower-bound heuristics
# ---------------------------------------------------------------------------

def lower_bound(G: nx.Graph) -> int:
    """Return a lower bound on Γ(G) via the reverse smallest-last strategy.

    Computes the smallest-last vertex ordering, reverses it, applies greedy
    coloring, and returns the number of colors used.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    int
        A lower bound on Γ(G).

    Complexity
    ----------
    O(n + m).
    """
    result = strategy_smallest_last(G)
    result.reverse()
    coloring = greedy_coloring(G, result)
    return max(coloring.values()) + 1


def lower_bound2(G: nx.Graph) -> int:
    """Return a lower bound on Γ(G) via the reverse largest-first strategy.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    int
        A lower bound on Γ(G).

    Complexity
    ----------
    O(n + m).
    """
    result = strategy_largest_first(G)
    result.reverse()
    coloring = greedy_coloring(G, result)
    return max(coloring.values()) + 1


# ---------------------------------------------------------------------------
# Maximal clique enumeration (Bron–Kerbosch with pivot, iterative)
# ---------------------------------------------------------------------------

def find_cliques(G: nx.Graph, nodes: Optional[list] = None):
    """Enumerate all maximal cliques of the undirected graph *G*.

    Uses the Bron–Kerbosch algorithm with Tomita–Tanaka–Takahashi pivot
    selection, implemented iteratively to avoid Python recursion-depth limits.

    Parameters
    ----------
    G : nx.Graph
        An undirected graph.
    nodes : list, optional
        If provided, only yield maximal cliques containing all nodes in this
        list.  ``nodes`` must form a clique; a ``ValueError`` is raised
        otherwise.

    Yields
    ------
    list
        Each yielded value is a list of nodes forming a maximal clique in *G*.

    Raises
    ------
    ValueError
        If the ``nodes`` argument does not form a clique in *G*.

    Complexity
    ----------
    O(3^{n/3}) in the worst case (Moon–Moser bound).

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
# Branch-and-bound solvers
# ---------------------------------------------------------------------------

def branch(G: nx.Graph) -> dict:
    """Compute the exact Grundy number of *G* using branch-and-bound with the
    **trivial upper bound** Δ(G[S]) + 1.

    Algorithm overview
    ------------------
    The search tree is built using the recurrence:

        Γ(S) = max{ Γ(S \\ X) + 1  |  X ⊆ S is an independent dominating set of G[S] }

    At each node of the search tree every maximal independent set X of the
    current subgraph G[S] is a candidate.  The tree is pruned whenever:

        |C| + upper_bound(G[S]) ≤ LB

    where |C| is the number of color classes already selected and LB is the
    current best lower bound (initialised by two greedy heuristics).

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    dict
        Keys: ``gamma`` (int), ``classes`` (list of list), ``valid`` (bool),
        ``cpu_s`` (float), ``bb_nodes`` (int), ``model`` (str).

    References
    ----------
    .. [BFK18] Bonnet et al. (2018), §3.
    """
    LB = max(lower_bound(G), lower_bound2(G))
    bestC: list[list] = []
    bb_nodes = 0

    def expand(G: nx.Graph, C: list[list]) -> None:
        nonlocal LB, bb_nodes, bestC
        if len(G) == 0:
            if len(C) > LB:
                LB = len(C)
                bestC = C
            return
        if len(C) + upper_bound(G) <= LB:
            return
        bb_nodes += 1
        Gc = nx.complement(G)
        for cor in find_cliques(Gc):
            newC = C + [cor]
            H = G.copy()
            H.remove_nodes_from(cor)
            expand(H, newC)

    start = time.time()
    expand(G, [])
    return {
        "model":    "BB",
        "gamma":    LB,
        "cpu_s":    time.time() - start,
        "classes":  bestC,
        "valid":    is_greedy_coloring(G, bestC),
        "bb_nodes": bb_nodes,
    }


def branch2(G: nx.Graph) -> dict:
    """Compute the exact Grundy number of *G* using branch-and-bound with the
    **stair-factor upper bound** (quadratic residue computation).

    Identical to :func:`branch` except that the pruning test uses
    :func:`stair_factor` instead of :func:`upper_bound`.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    dict
        Same keys as :func:`branch`.

    See Also
    --------
    branch  : Version with the trivial upper bound.
    branch3 : Version with the fast (O(n + m)) stair-factor bound.

    References
    ----------
    .. [BGK05] Shi et al. (2005), Theorem 3.
    """
    LB = max(lower_bound(G), lower_bound2(G))
    bestC: list[list] = []
    bb_nodes = 0

    def expand(G: nx.Graph, C: list[list]) -> None:
        nonlocal LB, bb_nodes, bestC
        if len(G) == 0:
            if len(C) > LB:
                LB = len(C)
                bestC = C
            return
        if len(C) + stair_factor(G) <= LB:
            return
        bb_nodes += 1
        Gc = nx.complement(G)
        for cor in find_cliques(Gc):
            newC = C + [cor]
            H = G.copy()
            H.remove_nodes_from(cor)
            expand(H, newC)

    start = time.time()
    expand(G, [])
    return {
        "model":    "BB/stair_factor",
        "gamma":    LB,
        "cpu_s":    time.time() - start,
        "classes":  bestC,
        "valid":    is_greedy_coloring(G, bestC),
        "bb_nodes": bb_nodes,
    }


def branch3(G: nx.Graph) -> dict:
    """Compute the exact Grundy number of *G* using branch-and-bound with the
    **fast stair-factor upper bound** (O(n + m) bucket-queue residue computation).

    Identical to :func:`branch2` except that the pruning test uses
    :func:`fast_stair_factor` instead of :func:`stair_factor`.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    dict
        Same keys as :func:`branch`.

    See Also
    --------
    branch  : Version with the trivial upper bound.
    branch2 : Version with the (slower) stair-factor bound.

    References
    ----------
    .. [BGK05] Shi et al. (2005), Theorem 3.
    """
    LB = max(lower_bound(G), lower_bound2(G))
    bestC: list[list] = []
    bb_nodes = 0

    def expand(G: nx.Graph, C: list[list]) -> None:
        nonlocal LB, bb_nodes, bestC
        if len(G) == 0:
            if len(C) > LB:
                LB = len(C)
                bestC = C
            return
        if len(C) + fast_stair_factor(G) <= LB:
            return
        bb_nodes += 1
        Gc = nx.complement(G)
        for cor in find_cliques(Gc):
            newC = C + [cor]
            H = G.copy()
            H.remove_nodes_from(cor)
            expand(H, newC)

    start = time.time()
    expand(G, [])
    return {
        "model":    "BB/fast_stair_factor",
        "gamma":    LB,
        "cpu_s":    time.time() - start,
        "classes":  bestC,
        "valid":    is_greedy_coloring(G, bestC),
        "bb_nodes": bb_nodes,
    }


# ---------------------------------------------------------------------------
# Brute-force enumerative baseline
# ---------------------------------------------------------------------------

def enumerate_grundy_colorings(G: nx.Graph) -> dict:
    """Find the Grundy number of *G* by exhaustive permutation enumeration.

    Tries every vertex ordering, applies greedy coloring, and keeps the
    coloring that uses the most colors.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    dict
        Same keys as :func:`branch`.

    Complexity
    ----------
    O(n! · (n + m)).  Intended only for small graphs (n ≤ 12).
    """
    best = 0
    C: list[list] = []
    start = time.time()
    for perm in itertools.permutations(G.nodes()):
        coloring = greedy_coloring(G, perm)
        num_colors = max(coloring.values()) + 1
        if num_colors > best:
            best = num_colors
            C = [[] for _ in range(num_colors)]
            for u in coloring.keys():
                C[coloring[u]].append(u)
    return {
        "model":   "Enumerative",
        "gamma":   best,
        "cpu_s":   time.time() - start,
        "classes": C,
        "valid":   True,
    }


def counting_grundy_colorings(G: nx.Graph) -> int:
    """Count distinct Grundy colorings of *G* by exhaustive permutation enumeration.

    Two permutations that produce the same color-class partition are counted
    only once.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    int
        The number of distinct Grundy colorings (as set-partitions).

    Complexity
    ----------
    O(n! · (n + m)).  Intended only for small graphs (n ≤ 12).
    """
    grundy_colorings: set = set()
    for perm in itertools.permutations(G.nodes()):
        coloring = greedy_coloring(G, perm)
        num_colors = max(coloring.values()) + 1
        C = [[] for _ in range(num_colors)]
        for u in coloring.keys():
            C[coloring[u]].append(u)
        color_partition = tuple(tuple(sorted(cls)) for cls in C)
        grundy_colorings.add(color_partition)
    return len(grundy_colorings)


# ---------------------------------------------------------------------------
# Enumerative algorithm for Grundy colorings (enumerate_orders)
# ---------------------------------------------------------------------------

def enumerate_orders(G: nx.Graph) -> set:
    """Enumerate all distinct Grundy colorings of *G* without full permutation
    search.

    The algorithm builds color-class partitions incrementally via backtracking,
    placing one vertex at a time (from a fixed ordering ``remaining``) plus any
    *pending* vertices that were displaced from an earlier class.  For each
    vertex *v* being placed and each existing color class *C[i]*, four moves
    are explored:

    1. **No conflict** – *v* has no neighbour in *C[i]*: append *v* to *C[i]*.
    2. **New class after i** – insert a singleton ``{v}`` immediately after
       *C[i]*.
    3. **Eviction** – *v* enters *C[i]*, and its neighbours in *C[i]* are
       evicted into a *pending* list with ``min_class = i + 1``, so that the
       backtracking re-inserts them into *C[i+1], C[i+2], …* (but never back
       into *C[i]* or earlier).  This is valid because the evicted vertices
       form a subset of the independent set *C[i]* and therefore remain
       mutually non-adjacent.
    4. **New class at end** – open a fresh color class ``{v}`` at the end of
       the current partition.

    Each completed partition is validated by :func:`is_greedy_coloring` before
    being added to the output set.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph with n vertices and m edges.

    Returns
    -------
    set of tuple of tuple
        Each element is a tuple of sorted tuples representing a distinct Grundy
        coloring: ``((v1, v2, …), (v3, …), …)`` where the *i*-th inner tuple
        is color class *i*.

    Notes
    -----
    The ``pending`` list carries pairs ``(vertex, min_class)`` to ensure that
    evicted vertices are only re-inserted into classes strictly after the one
    from which they were removed.

    Complexity
    ----------
    Worst-case super-exponential (the number of candidate partitions can be
    large), but in practice far smaller than the n! brute-force baseline thanks
    to structural pruning via ``stair_factor``.
    """
    n = len(G)
    grundy_colorings: set = set()
    total_colorings = 0
    ub = stair_factor(G)

    def backtrack(
        C: list[list],
        remaining: list,
        idx: int,
        pending: list,          # list of (vertex, min_class)
    ) -> None:
        nonlocal total_colorings

        # Determine the current vertex to place.
        if pending:
            v, min_class = pending[0]
            rest_pending = pending[1:]
            advancing = False
        elif idx < n:
            v = remaining[idx]
            min_class = 0           # normal vertex: may enter any class
            rest_pending = []
            advancing = True
        else:
            # All vertices placed and no pending evictions: record result.
            total_colorings += 1
            if is_greedy_coloring(G, C):
                color_partition = tuple(tuple(sorted(cls)) for cls in C)
                grundy_colorings.add(color_partition)
            return

        next_idx = idx + 1 if advancing else idx

        # Pruning: current number of colors already exceeds upper bound.
        if len(C) > ub:
            return

        # Base case: no color class exists yet.
        if len(C) == 0:
            C.append([v])
            backtrack(C, remaining, next_idx, rest_pending)
            C.pop()
            return

        # Try inserting v into each existing class, starting from min_class.
        for i in range(min_class, len(C)):
            classe = C[i]
            vizinhos = [u for u in classe if u in G[v]]

            if len(vizinhos) == 0:
                # Move 1: no conflict – v joins class i.
                classe.append(v)
                backtrack(C, remaining, next_idx, rest_pending)
                classe.pop()

            else:
                # Move 2: open a new singleton class immediately after i.
                C.insert(i + 1, [v])
                backtrack(C, remaining, next_idx, rest_pending)
                C.pop(i + 1)

                # Move 3: v enters class i; conflicting neighbours are evicted
                # into pending with min_class = i + 1.
                # Correctness: evicted vertices ⊆ C[i] (an independent set),
                # so any subset of them is also independent and can form a
                # valid class at any position j > i.
                originais = list(classe)
                classe[:] = [u for u in classe if u not in vizinhos]
                classe.append(v)
                new_pending = [(u, i + 1) for u in vizinhos] + list(rest_pending)
                backtrack(C, remaining, next_idx, new_pending)
                classe[:] = originais

        # Move 4: open a fresh class at the very end of C.
        C.append([v])
        backtrack(C, remaining, next_idx, rest_pending)
        C.pop()

    start = time.time()
    backtrack([], list(G.nodes()), 0, [])
    elapsed = time.time() - start

    print(f"Distinct Grundy colorings : {len(grundy_colorings)}")
    print(f"Total partitions explored : {total_colorings}")
    print(f"Time                      : {elapsed:.4f}s")
    return grundy_colorings


# ---------------------------------------------------------------------------
# Correctness and performance test harnesses
# ---------------------------------------------------------------------------

def run_correctness_tests(solvers: list) -> list:
    """Test solvers on instances with known Grundy numbers.

    Parameters
    ----------
    solvers : list of (str, callable)
        Each entry is ``(name, solver_function)``.  The solver must accept a
        single ``nx.Graph`` argument and return a dict with at least the keys
        ``gamma`` (int) and ``valid`` (bool).

    Returns
    -------
    list
        Raw results for further inspection.
    """
    tests = [
        ("Path P4",        nx.path_graph(4),                  3),
        ("Path P5",        nx.path_graph(5),                  3),
        ("Path P6",        nx.path_graph(6),                  3),
        ("Path P7",        nx.path_graph(7),                  3),
        ("Cycle C4",       nx.cycle_graph(4),                 2),
        ("Cycle C5",       nx.cycle_graph(5),                 3),
        ("Cycle C6",       nx.cycle_graph(6),                 3),
        ("Cycle C7",       nx.cycle_graph(7),                 3),
        ("Cycle C8",       nx.cycle_graph(8),                 3),
        ("Complete K3",    nx.complete_graph(3),              3),
        ("Complete K4",    nx.complete_graph(4),              4),
        ("Complete K5",    nx.complete_graph(5),              5),
        ("Complete K6",    nx.complete_graph(6),              6),
        ("Bipartite K33",  nx.complete_bipartite_graph(3, 3), 2),
        ("ER n=8 p=0.1",   nx.erdos_renyi_graph(8,  0.1, seed=1), None),
        ("ER n=8 p=0.3",   nx.erdos_renyi_graph(8,  0.3, seed=1), None),
        ("ER n=8 p=0.5",   nx.erdos_renyi_graph(8,  0.5, seed=1), None),
        ("ER n=8 p=0.7",   nx.erdos_renyi_graph(8,  0.7, seed=1), None),
        ("ER n=8 p=0.9",   nx.erdos_renyi_graph(8,  0.9, seed=1), None),
    ]

    W = 26
    model_names = [name for name, _ in solvers]
    hdr = (f"{'Graph':<{W}} {'Γ':>4} "
           + " ".join(f"{m:>10}" for m in model_names)
           + f" {'Valid':>6} {'OK':>4}")
    print(hdr)
    print("─" * len(hdr))

    total, passed = 0, 0
    all_results = []

    for name, G, chi_exp in tests:
        results, chis, valids = [], [], []
        for _, solver in solvers:
            r = solver(G)
            results.append(r)
            chis.append(r["gamma"])
            valids.append(r["valid"])

        agree      = all(c == chis[0] for c in chis)
        correct    = (chi_exp is None) or (chis[0] == chi_exp)
        valid_all  = all(valids)
        ok         = agree and correct and valid_all
        total += 1
        if ok:
            passed += 1

        exp_s   = str(chi_exp) if chi_exp else "?"
        chi_str = " ".join(f"{c:>10}" for c in chis)
        print(f"{name:<{W}} {exp_s:>4} {chi_str} {'✓' if valid_all else '✗':>6} {'✓' if ok else '✗':>4}")
        all_results.append((name, G, chi_exp, results))

    print("─" * len(hdr))
    print(f"Result: {passed}/{total} tests passed\n")
    return all_results


def run_performance_tests(solvers: list) -> None:
    """Benchmark solvers on a set of larger instances.

    Parameters
    ----------
    solvers : list of (str, callable)
        Same format as :func:`run_correctness_tests`.
    """
    perf = [
        ("Petersen",       nx.petersen_graph()),
        ("Hypercube Q4",   nx.hypercube_graph(4)),
        ("Cycle C20",      nx.cycle_graph(20)),
        ("Mycielski χ=3",  nx.mycielski_graph(3)),
        ("Mycielski χ=4",  nx.mycielski_graph(4)),
        ("Mycielski χ=5",  nx.mycielski_graph(5)),
        ("ER n=15 p=0.1",  nx.erdos_renyi_graph(15, 0.1, seed=1)),
        ("ER n=15 p=0.3",  nx.erdos_renyi_graph(15, 0.3, seed=3)),
        ("ER n=15 p=0.5",  nx.erdos_renyi_graph(15, 0.5, seed=5)),
        ("ER n=15 p=0.7",  nx.erdos_renyi_graph(15, 0.7, seed=7)),
        ("ER n=15 p=0.9",  nx.erdos_renyi_graph(15, 0.9, seed=9)),
    ]

    W = 22
    model_names = [name for name, _ in solvers]
    hdr = (f"{'Graph':<{W}} {'Γ':>4} "
           + " ".join(f"{m:>12}" for m in model_names))
    print(hdr)
    print("─" * len(hdr))

    for name, G in perf:
        results = [solver(G) for _, solver in solvers]
        gamma   = results[0]["gamma"]
        times   = " ".join(f"{r['cpu_s']:>11.3f}s" for r in results)
        print(f"{name:<{W}} {gamma:>4} {times}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    solvers = [
        ("BB",      branch),
        ("BB/SF",   branch2),
        ("BB/FSF",  branch3),
    ]

    G = nx.erdos_renyi_graph(15, 0.2, seed=42)

    print("=== Correctness tests ===")
    run_correctness_tests(solvers)

    print("=== Performance tests ===")
    run_performance_tests(solvers)

    print("=== Enumerate orders (small graph) ===")
    G_small = nx.erdos_renyi_graph(8, 0.2, seed=1)
    bf_count = counting_grundy_colorings(G_small)
    eo_set   = enumerate_orders(G_small)
    print(f"Brute-force count : {bf_count}")
    print(f"enumerate_orders  : {len(eo_set)}")
    print(f"Match             : {bf_count == len(eo_set)}")
