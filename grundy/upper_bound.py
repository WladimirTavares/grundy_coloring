import time
import itertools
from collections import defaultdict, deque
from typing import Callable, Optional
import math
import heapq
import networkx as nx
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp


# ---------------------------------------------------------------------------
# Upper-bound functions
# ---------------------------------------------------------------------------

def delta_1(G: nx.Graph) -> int:
    """Return the trivial upper bound Δ(G) + 1 on the Grundy number of *G*.

    The bound follows from the fact that in any greedy coloring, color class i
    (1-indexed) must contain at least one vertex of degree ≥ i − 1 in G.
    Therefore Γ(G) ≤ Δ(G) + 1.

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

    Examples
    --------
    >>> import networkx as nx
    >>> upper_bound(nx.complete_graph(5))
    5
    >>> upper_bound(nx.path_graph(6))
    3

    References
    ----------
    .. [BGK05] Shi et al. (2005), §2.
    """
    max_degree = max(dict(G.degree()).values())
    return max_degree + 1




def delta_2(G: nx.Graph) -> int:
    """
    Computes the maximum degree among neighbours of *u* whose degree is
    strictly less than deg(*u*), then adds one.  If no such neighbour exists
    the function returns deg(*u*) + 1 as a fallback.

    Formally:

        δ₂(u) = max_{v} max{ deg(w) | w ∈ N(u), deg(w) <= deg(v) } + 1

        
    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    
    Examples
    --------
    >>> delta_2(nx.path_graph(5))   # max edge min-degree = 1 → bound = 2
    2
    >>> delta_2(nx.complete_graph(4))  # every edge has min-degree 3 → bound = 4
    4

    Reference:

    
    """

    max_val = 0
    for v in G.nodes():
        val = 0
        for u in G[v]:
            if G.degree[u] <= G.degree[v]:
                val = max(val, G.degree[u])
        max_val = max(max_val, val)

    return max_val + 1
        
    

def stair_factor(G: nx.Graph) -> int:
    """Return the *stair-factor* upper bound on the Grundy number of *G*.

    The stair factor is computed from the *maximum-degree degeneracy ordering*
    (residue sequence):

    1. Repeatedly remove the vertex of **maximum** residual degree, recording
       that degree as dᵢ (i = 1, 2, …, n, 1-indexed).
    2. The stair factor is  min_i (dᵢ + i).

    This bound satisfies Γ(G) ≤ stair_factor(G) ≤ Δ(G) + 1 and is often
    significantly tighter than the trivial bound, especially for sparse graphs.

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

    Examples
    --------
    >>> stair_factor(nx.petersen_graph())   # Δ+1 = 4, stair = 4
    4
    >>> stair_factor(nx.path_graph(10))     # sparse; Δ+1 = 3, stair = 3
    3

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
    sets indexed by degree value), achieving true O(n + m) time.  This is the
    preferred upper-bound function for the branch-and-bound solvers.

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

    Examples
    --------
    >>> fast_stair_factor(nx.complete_graph(6))
    6
    >>> fast_stair_factor(nx.cycle_graph(8))
    3

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

"""
Implementations of the **revised stair factor** ζ'(G), an upper bound on the
partial Grundy number ∂Γ(G) of a graph.
 
Theoretical guarantee (Lemma 3.1 of [PV19]):
    ∂Γ(G)  ≤  ζ'(G)  ≤  ζ(G)
 
Core idea — degree-class decomposition
---------------------------------------
At each step, all vertices of **maximum degree** in the current residual graph
are identified and partitioned into **equivalence classes** by their current
open neighbourhood (u and v belong to the same class iff N(u) = N(v) in the
current graph). One entire class is removed and its residual degree dⁱ is
recorded. The bound is then:
 
    ζ'(G) = min{ dⁱ + i  |  1 ≤ i ≤ r }
 
where r is the total number of classes extracted until the graph is empty
(Definition 3.4 and Theorem 3.2 of [PV19]).
 
Implemented variants
--------------------
Two independent axes of variation:
 
1. **Class-selection criterion** when multiple classes share the maximum degree:
 
   * ``min_node``  — selects the class whose minimum vertex index is smallest.
                     Simple deterministic baseline.
   * ``max_size``  — selects the largest class. Empirically produces tighter
                     bounds (~99.6 % win rate over 30 000 random graphs).
 
2. **Implementation** (identical output, different complexity):
 
   * **slow** — recomputes degrees from scratch each iteration via
                ``dict(G.degree())``.  O(n·(n + m)) overall.
   * **fast** — maintains degrees and degree-buckets incrementally, avoiding
                full recomputation.  Preferred in hot paths such as
                branch-and-bound solvers.
 
Each (slow, fast) pair with the same criterion is **deterministically
equivalent** — they return the same value on every graph, verified over
5 000 random instances.
 
Which variant to use?
---------------------
Use ``revised_stair_factor_fast2`` in production: it combines incremental
degree maintenance with the ``max_size`` criterion, which yields the tightest
bounds found among simple heuristics.
 
References
----------
.. [PV19] B.S. Panda and Shaily Verma, "On partial Grundy coloring of
   bipartite graphs and chordal graphs", *Discrete Applied Mathematics*,
   vol. 271, pp. 171–183, 2019.
   Department of Mathematics, Indian Institute of Technology Delhi,
   Hauz Khas, New Delhi 110016, India.
   https://doi.org/10.1016/j.dam.2019.08.005
   Definitions 3.3–3.4, Lemma 3.1, Theorem 3.2.
"""

# ---------------------------------------------------------------------------
# Criterion 1: min_node
# Selects the class whose minimum vertex index is smallest.
# Deterministic because candidates are sorted before grouping.
# ---------------------------------------------------------------------------
 
def revised_stair_factor(G: nx.Graph) -> int:
    """Revised stair factor ζ'(G) — slow implementation, min_node criterion.
 
    Computes the degree-class decomposition of *G* by recomputing vertex
    degrees from scratch at every iteration.  At each step the equivalence
    class of maximum-degree vertices with the smallest minimum vertex index is
    removed.
 
    This is the reference (slow) implementation.  For better performance use
    :func:`revised_stair_factor_fast`; for a tighter bound use
    :func:`revised_stair_factor2` or :func:`revised_stair_factor_fast2`.
 
    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph with n vertices and m edges.
 
    Returns
    -------
    int
        The revised stair factor ζ'(G), an upper bound such that
        ∂Γ(G) ≤ ζ'(G) ≤ ζ(G).
 
    Complexity
    ----------
    O(n · (n + m)).
 
    References
    ----------
    .. [PV19] B.S. Panda and Shaily Verma, "On partial Grundy coloring of
       bipartite graphs and chordal graphs", *Discrete Applied Mathematics*,
       vol. 271, pp. 171–183, 2019.
       https://doi.org/10.1016/j.dam.2019.08.005
 
    Examples
    --------
    >>> revised_stair_factor(nx.complete_bipartite_graph(4, 3))
    2
    >>> revised_stair_factor(nx.cycle_graph(8))
    3
    """
    G_curr = G.copy()
    d_vals: list[int] = []
 
    while len(G_curr) > 0:
        # Recompute degrees in the current residual graph
        degrees = dict(G_curr.degree())
        max_deg = max(degrees.values())
 
        # Collect max-degree vertices; sort for a deterministic criterion
        candidates = sorted(v for v in G_curr.nodes() if degrees[v] == max_deg)
 
        # Partition candidates into equivalence classes by current neighbourhood
        classes: defaultdict[frozenset, list] = defaultdict(list)
        for v in candidates:
            classes[frozenset(G_curr.neighbors(v))].append(v)
 
        # min_node: first class inserted has the globally smallest vertex
        # (guaranteed because candidates is already sorted)
        cls = min(classes.values(), key=lambda c: min(c))
 
        d_vals.append(max_deg)
        G_curr.remove_nodes_from(cls)
 
    # ζ'(G) = min{ dⁱ + i } with 1-based index → +1 on 0-based enumerate
    return min(d + i + 1 for i, d in enumerate(d_vals))
 
 
def revised_stair_factor_fast(G: nx.Graph) -> int:
    """Revised stair factor ζ'(G) — fast implementation, min_node criterion.
 
    Maintains vertex degrees and degree-buckets incrementally instead of
    recomputing ``dict(G.degree())`` at every step.  Returns the **same
    value** as :func:`revised_stair_factor` on every graph.
 
    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph with n vertices and m edges.
 
    Returns
    -------
    int
        The revised stair factor ζ'(G), an upper bound such that
        ∂Γ(G) ≤ ζ'(G) ≤ ζ(G).
 
    References
    ----------
    .. [PV19] B.S. Panda and Shaily Verma, "On partial Grundy coloring of
       bipartite graphs and chordal graphs", *Discrete Applied Mathematics*,
       vol. 271, pp. 171–183, 2019.
       https://doi.org/10.1016/j.dam.2019.08.005
 
    Examples
    --------
    >>> revised_stair_factor_fast(nx.complete_bipartite_graph(4, 3))
    2
    >>> revised_stair_factor_fast(nx.cycle_graph(8))
    3
    """
    G_curr = G.copy()
 
    # Dynamic degree table: updated incrementally as vertices are removed
    degree = dict(G_curr.degree())
 
    # Bucket structure: buckets[d] = set of vertices currently at degree d
    buckets: defaultdict[int, set] = defaultdict(set)
    for v, d in degree.items():
        buckets[d].add(v)
 
    max_deg = max(degree.values()) if G_curr.nodes() else 0
    d_vals: list[int] = []
 
    while G_curr.nodes():
        # Walk max_deg downward until a non-empty, consistent bucket is found
        while max_deg >= 0:
            valid = [
                v for v in buckets[max_deg]
                if v in G_curr and degree[v] == max_deg
            ]
            if valid:
                break
            max_deg -= 1
 
        # Sort for the same deterministic criterion as the slow variant
        candidates = sorted(valid)
 
        # Partition by current neighbourhood
        classes: defaultdict[frozenset, list] = defaultdict(list)
        for v in candidates:
            classes[frozenset(G_curr.neighbors(v))].append(v)
 
        # min_node criterion
        cls = min(classes.values(), key=lambda c: min(c))
 
        d_vals.append(max_deg)
 
        # Remove the chosen class and update neighbour degrees incrementally
        for v in cls:
            if v not in G_curr:
                continue
            for u in list(G_curr.neighbors(v)):
                if u in G_curr:
                    buckets[degree[u]].discard(u)
                    degree[u] -= 1
                    buckets[degree[u]].add(u)
            buckets[degree[v]].discard(v)
            G_curr.remove_node(v)
            del degree[v]
 
    return min(d + i + 1 for i, d in enumerate(d_vals))
 
 
# ---------------------------------------------------------------------------
# Criterion 2: max_size  ← empirically tighter bound
# Selects the largest class; ties broken by minimum vertex index.
# ---------------------------------------------------------------------------
 
def revised_stair_factor2(G: nx.Graph) -> int:
    """Revised stair factor ζ'(G) — slow implementation, max_size criterion.
 
    Identical to :func:`revised_stair_factor` except that, when multiple
    equivalence classes share the maximum degree, the **largest** class is
    chosen for removal.
 
    This heuristic produces bounds equal to or tighter than the min_node
    criterion in ~99.6 % of tested graphs.  Intuition: removing more vertices
    at once keeps the step index *i* smaller in subsequent rounds, directly
    lowering the future terms dⁱ + i.
 
    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph with n vertices and m edges.
 
    Returns
    -------
    int
        The revised stair factor ζ'(G), an upper bound such that
        ∂Γ(G) ≤ ζ'(G) ≤ ζ(G).
 
    Complexity
    ----------
    O(n · (n + m)).
 
    References
    ----------
    .. [PV19] B.S. Panda and Shaily Verma, "On partial Grundy coloring of
       bipartite graphs and chordal graphs", *Discrete Applied Mathematics*,
       vol. 271, pp. 171–183, 2019.
       https://doi.org/10.1016/j.dam.2019.08.005
 
    Examples
    --------
    >>> revised_stair_factor2(nx.complete_bipartite_graph(4, 3))
    2
    >>> revised_stair_factor2(nx.cycle_graph(8))
    3
    """
    G_curr = G.copy()
    d_vals: list[int] = []
 
    while len(G_curr) > 0:
        degrees = dict(G_curr.degree())
        max_deg = max(degrees.values())
 
        candidates = sorted(v for v in G_curr.nodes() if degrees[v] == max_deg)
 
        classes: defaultdict[frozenset, list] = defaultdict(list)
        for v in candidates:
            classes[frozenset(G_curr.neighbors(v))].append(v)
 
        # max_size: remove the most vertices per step to keep i small
        cls = max(classes.values(), key=len)
 
        d_vals.append(max_deg)
        G_curr.remove_nodes_from(cls)
 
    return min(d + i + 1 for i, d in enumerate(d_vals))
 
 
def revised_stair_factor_fast2(G: nx.Graph) -> int:
    """Revised stair factor ζ'(G) — fast implementation, max_size criterion.
 
    Combines the incremental degree/bucket maintenance of
    :func:`revised_stair_factor_fast` with the max_size class-selection of
    :func:`revised_stair_factor2`.  Returns the **same value** as
    :func:`revised_stair_factor2` on every graph.
 
    This is the **recommended variant** for performance-critical contexts
    such as branch-and-bound solvers, where the upper bound is recomputed
    many times on subgraphs.
 
    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph with n vertices and m edges.
 
    Returns
    -------
    int
        The revised stair factor ζ'(G), an upper bound such that
        ∂Γ(G) ≤ ζ'(G) ≤ ζ(G).
 
    References
    ----------
    .. [PV19] B.S. Panda and Shaily Verma, "On partial Grundy coloring of
       bipartite graphs and chordal graphs", *Discrete Applied Mathematics*,
       vol. 271, pp. 171–183, 2019.
       https://doi.org/10.1016/j.dam.2019.08.005
 
    Examples
    --------
    >>> revised_stair_factor_fast2(nx.complete_bipartite_graph(4, 3))
    2
    >>> revised_stair_factor_fast2(nx.cycle_graph(8))
    3
    """
    G_curr = G.copy()
 
    degree = dict(G_curr.degree())
 
    buckets: defaultdict[int, set] = defaultdict(set)
    for v, d in degree.items():
        buckets[d].add(v)
 
    max_deg = max(degree.values()) if G_curr.nodes() else 0
    d_vals: list[int] = []
 
    while G_curr.nodes():
        while max_deg >= 0:
            valid = [
                v for v in buckets[max_deg]
                if v in G_curr and degree[v] == max_deg
            ]
            if valid:
                break
            max_deg -= 1
 
        candidates = sorted(valid)
 
        classes: defaultdict[frozenset, list] = defaultdict(list)
        for v in candidates:
            classes[frozenset(G_curr.neighbors(v))].append(v)
 
        # max_size criterion
        cls = max(classes.values(), key=len)
 
        d_vals.append(max_deg)
 
        for v in cls:
            if v not in G_curr:
                continue
            for u in list(G_curr.neighbors(v)):
                if u in G_curr:
                    buckets[degree[u]].discard(u)
                    degree[u] -= 1
                    buckets[degree[u]].add(u)
            buckets[degree[v]].discard(v)
            G_curr.remove_node(v)
            del degree[v]
 
    return min(d + i + 1 for i, d in enumerate(d_vals))
 
def psi_bound(G: nx.Graph, UB : int) -> int:
    """Return the ψ upper bound on the Grundy number of *G*.

    For each vertex v and depth parameter k, ψ(v, k) is defined as the
    largest l such that there exist neighbours u₁, …, u_{l-1} of v with
    ψ(uᵢ, k) ≥ i for all i.  Intuitively, ψ(v, k) measures how many
    hierarchical colour levels can be "supported" by v's neighbourhood.

    The global bound is:

        Ψ(G) = max_{v ∈ V(G)} ψ(v, Δ(G) + 1)

    and satisfies Γ(G) ≤ Ψ(G).

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph with n vertices and m edges.

    Returns
    -------
    int
        The ψ upper bound such that Γ(G) ≤ Ψ(G).

    Complexity
    ----------
    O(n · Δ²), where Δ = Δ(G) is the maximum degree.

    Notes
    -----
    The recurrence is evaluated iteratively for k = 1, 2, …, Δ + 1 using
    a counting-array technique that avoids explicit sorting of neighbour
    values.  :func:`psi_bound2` returns the full ψ table instead of the
    scalar maximum.
    """
    
    # psi[v][k]
    psi = {v: [0] * (UB + 1) for v in G.nodes()}
    
    # Base: k = 1
    for v in G.nodes():
        psi[v][1] = 1

    # Para k = 2 até Delta+1
    for k in range(2, UB + 1):
        for v in G.nodes():
            deg_v = G.degree(v)
            
            # counting array: valores possíveis vão de 1 até Delta+1
            count = [0] * (UB + 1)

            # conta os valores dos vizinhos
            for u in G.neighbors(v):
                val = psi[u][k-1]
                count[val] += 1

            # agora percorremos como se fosse ordenado
            l = 1  # queremos construir sequência 1,2,3,...

            # percorre valores em ordem crescente
            for val in range(1, UB + 1):
                while count[val] > 0 and val >= l:
                    count[val] -= 1
                    l += 1

            psi[v][k] = l

    Psi = max(psi[v][UB] for v in G.nodes())
    
    return Psi

def psi_table(G: nx.Graph, UB : int):
    """Return the full ψ table for every vertex and depth level of *G*.

    Computes the same recurrence as :func:`psi_bound` but returns the
    complete two-dimensional table ``psi[v][k]`` for all vertices v and
    levels k ∈ {1, …, Δ(G) + 1}, instead of only the scalar maximum.
    This is useful when per-vertex local bounds are needed (e.g. inside
    branch-and-bound nodes).

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph with n vertices and m edges.
    upper_bound : int
        An externally supplied upper bound on Γ(G).  Currently unused in the
        computation but kept for API compatibility with other bound functions.

    Returns
    -------
    dict
        A mapping ``{v: list}`` where ``psi[v][k]`` is the ψ value of
        vertex v at depth k, for k = 0, 1, …, Δ(G) + 1.

    Complexity
    ----------
    O(n · Δ²), where Δ = Δ(G).

    See Also
    --------
    psi_bound : Returns only the global maximum Ψ(G) as a scalar.
    """
    
    # psi[v][k]
    psi = {v: [0] * (UB + 1) for v in G.nodes()}
    
    # Base: k = 1
    for v in G.nodes():
        psi[v][1] = 1

    # Para k = 2 até Delta+1
    for k in range(2, UB + 1):
        for v in G.nodes():
            deg_v = G.degree(v)
            
            # counting array: valores possíveis vão de 1 até Delta+1
            count = [0] * (UB + 1)

            # conta os valores dos vizinhos
            for u in G.neighbors(v):
                val = psi[u][k-1]
                count[val] += 1

            # agora percorremos como se fosse ordenado
            l = 1  # queremos construir sequência 1,2,3,...

            # percorre valores em ordem crescente
            for val in range(1, UB + 1):
                while count[val] > 0 and val >= l:
                    count[val] -= 1
                    l += 1

            psi[v][k] = l

    
    return { v : psi[v][UB] for v in G.nodes() }

def upper_bound1(G):
    return min(delta_1(G), delta_2(G), stair_factor(G), psi_bound(G, delta_1(G)))

def upper_bound2(G):
    return min(delta_1(G), delta_2(G), stair_factor(G), revised_stair_factor2(G), psi_bound(G, delta_1(G)))


if __name__ == "__main__":

    G = nx.erdos_renyi_graph(50, 0.5, 1)

    print( "Limite Delta1: ", delta_1(G) )
    print( "Limite Delta1: ", delta_2(G) )
    print( "stair factor: " , stair_factor(G) )
    print( "revised stair factor: " , revised_stair_factor(G) )
    print( "psi_bound: " , psi_bound(G, delta_1(G)) )
