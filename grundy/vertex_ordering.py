import networkx as nx
import itertools
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Vertex ordering strategies
# ---------------------------------------------------------------------------

def largest_first_ordering(G: nx.Graph) -> list:
    """Return vertices of *G* ordered by decreasing residual degree (largest-first).

    At each step the vertex with the highest current residual degree is
    selected, appended to the result, and removed from the working graph;
    the residual degrees of its neighbours are decremented.

    This ordering tends to produce dense early color classes when used as input
    to :func:`greedy_coloring`, which often yields a good lower bound on Γ(G).

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    list
        Vertices in decreasing-residual-degree order.  The first element had
        the highest residual degree at its time of removal.

    Complexity
    ----------
    O(n + m) using a bucket-queue of size Δ(G).

    See Also
    --------
    strategy_smallest_last : Complementary ordering strategy.
    lower_bound2            : Uses the reverse of this ordering as a lower bound.
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


def smallest_last_ordering(G: nx.Graph) -> list:
    """Return vertices of *G* ordered so that the vertex of minimum residual
    degree is placed **last** (smallest-last / degeneracy ordering).

    The ordering is constructed by iteratively removing the vertex of minimum
    residual degree and prepending it to the result.  When reversed, this
    ordering is equivalent to the degeneracy ordering and tends to produce a
    high greedy-coloring lower bound on Γ(G).

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

    See Also
    --------
    strategy_largest_first : Complementary ordering strategy.
    lower_bound            : Uses the reverse of this ordering as a lower bound.
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


def large_clique(G):
    
    degrees = dict(G.degree())
    
    best_clique = []

    def _clique_heuristic(C, U):
        nonlocal best_clique

        if not U:
            if len(C) > len(best_clique):
                best_clique = C.copy()
            return
        
        # escolha gulosa: vértice de maior grau
        u = max(U, key=lambda x: degrees[x])
        
        U = U - {u}
        C.append(u)

        # poda: só vizinhos com grau suficiente
        N_prime = {v for v in G[u] if v in U and degrees[v] >= len(best_clique)}

        _clique_heuristic(C, N_prime)

        # backtrack
        C.pop()

    for u in G:
        if degrees[u] < len(best_clique):
            continue
        
        neighbors = {v for v in G[u] if degrees[v] >= len(best_clique)}
        _clique_heuristic([u], neighbors)

    return best_clique


def clique_order_smallest_order(G: nx.Graph) -> list:
    clique = large_clique(G)
    order = clique
    H = G.copy()
    for v in clique:
        H.remove_node(v)
    order.extend( smallest_last_ordering(H) )
    return order




    
    
def reverse_smallest_last(G: nx.Graph) -> list:
    """Return the reverse of the smallest-last (degeneracy) ordering of *G*.

    Computes the smallest-last ordering via :func:`strategy_smallest_last`
    and reverses it in place.  The resulting order processes vertices from
    the one removed last (highest residual degree) to the one removed first
    (lowest residual degree), which is the standard degeneracy ordering used
    as input to :func:`lower_bound`.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    list
        Vertices in reverse smallest-last order.

    Complexity
    ----------
    O(n + m).

    See Also
    --------
    strategy_smallest_last : Forward ordering (smallest removed last).
    lower_bound            : Uses this ordering to initialise Γ(G).
    """
    nodes = smallest_last_ordering(G)
    nodes.reverse()
    return nodes


def reverse_largest_first(G: nx.Graph) -> list:
    """Return the reverse of the smallest-last (degeneracy) ordering of *G*.

    Computes the smallest-last ordering via :func:`strategy_smallest_last`
    and reverses it in place.  The resulting order processes vertices from
    the one removed last (highest residual degree) to the one removed first
    (lowest residual degree), which is the standard degeneracy ordering used
    as input to :func:`lower_bound`.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    list
        Vertices in reverse smallest-last order.

    Complexity
    ----------
    O(n + m).

    See Also
    --------
    strategy_smallest_last : Forward ordering (smallest removed last).
    lower_bound            : Uses this ordering to initialise Γ(G).
    """
    nodes = smallest_last_ordering(G)
    nodes.reverse()
    return nodes





if __name__ == "__main__":

    G = nx.erdos_renyi_graph(10, 0.5)

    print( smallest_last_ordering(G) )