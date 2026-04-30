import networkx as nx

# ---------------------------------------------------------------------------
# Greedy coloring
# ---------------------------------------------------------------------------

def greedy_coloring(G: nx.Graph, nodes: list) -> dict:
    """Assign colors to vertices of *G* greedily in the given vertex order.

    Each vertex is assigned the smallest non-negative integer color (the
    *minimum excludant*, or mex) not already used by any of its
    already-colored neighbours.  This is the first-fit coloring rule.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.
    nodes : list
        An ordering of *all* vertices of *G*.  Every vertex must appear
        exactly once.

    Returns
    -------
    dict
        A mapping ``{vertex: color}`` where colors are non-negative integers
        starting from 0.

    Complexity
    ----------
    O(n + m), where n = |V| and m = |E|.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> greedy_coloring(G, [0, 1, 2, 3])
    {0: 0, 1: 1, 2: 0, 3: 1}
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

    A coloring C = (C₀, C₁, …, C_{k-1}) is *greedy* if and only if:

    1. Each Cᵢ is an independent set (no two vertices in Cᵢ are adjacent).
    2. Every vertex v ∈ Cᵢ has at least one neighbour in each earlier class
       Cⱼ for j < i  (the *Grundy property*).

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.
    C : list of list
        A partition of V(G) into color classes ordered C₀, C₁, …, C_{k-1}.

    Returns
    -------
    bool
        ``True`` if both conditions hold; ``False`` otherwise.

    Complexity
    ----------
    O(k · n · Δ) in the worst case.

    Examples
    --------
    >>> G = nx.path_graph(4)          # 0-1-2-3
    >>> is_greedy_coloring(G, [[0, 2], [1, 3]])
    True
    >>> is_greedy_coloring(G, [[1, 3], [0, 2]])
    False
    """
    if not C:
        return True
    sets = [set(cls) for cls in C]
    nbr  = {v: set(G[v]) for v in G.nodes()}
    for i, cls in enumerate(sets):
        for v in cls:
            if nbr[v] & cls:              # condição 1: independência
                return False
            for j in range(i):            # condição 2: Grundy
                if not (nbr[v] & sets[j]):
                    return False
    return True


from vertex_ordering import reverse_smallest_last


if __name__ == "__main__":

    
    G = nx.erdos_renyi_graph(50, 0.5)    
    print(greedy_coloring(G, reverse_smallest_last(G)) )