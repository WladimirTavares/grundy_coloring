import networkx as nx

from greedy_coloring import greedy_coloring
from vertex_ordering import smallest_last_ordering
from vertex_ordering import largest_first_ordering

def lb_reverse_sl(G: nx.Graph) -> dict:
    """Return a lower bound on Γ(G) via the reverse smallest-last strategy.

    Computes the smallest-last vertex ordering (degeneracy ordering), reverses
    it, applies :func:`greedy_coloring`, and returns the number of colors used.
    The result is a valid Grundy coloring because any greedy coloring is one.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    dict
        ``{"coloring": list[list], "lower_bound": int}``

        * **coloring** – the Grundy coloring found, as a list of color classes.
        * **lower_bound** – number of colors used (= a valid lower bound on Γ(G)).

    Complexity
    ----------
    O(n + m).

    See Also
    --------
    lower_bound2            : Uses the reverse largest-first ordering instead.
    strategy_smallest_last  : The ordering subroutine.
    """
    result = smallest_last_ordering(G)
    result.reverse()
    coloring = greedy_coloring(G, result)
    bound = max(coloring.values()) + 1
    C = [[] for _ in range(bound)]
    for u in G.nodes():
        C[coloring[u]].append(u)
    return {"coloring": C, "lower_bound": bound}


def lb_reverse_lf(G: nx.Graph) -> dict:
    """Return a lower bound on Γ(G) via the reverse largest-first strategy.

    Computes the largest-first vertex ordering (by residual degree), reverses
    it, applies :func:`greedy_coloring`, and returns the number of colors used.

    This heuristic complements :func:`lower_bound`: on dense graphs the
    largest-first ordering tends to produce more colors, while on sparse graphs
    the smallest-last ordering is typically better.  Both are used together by
    the branch-and-bound solvers to initialise their lower bound LB.

    Parameters
    ----------
    G : nx.Graph
        A simple undirected graph.

    Returns
    -------
    dict
        ``{"coloring": list[list], "lower_bound": int}``

        * **coloring** – the Grundy coloring found.
        * **lower_bound** – number of colors used.

    Complexity
    ----------
    O(n + m).

    See Also
    --------
    lower_bound            : Uses the reverse smallest-last ordering instead.
    strategy_largest_first : The ordering subroutine.
    """
    result = largest_first_ordering(G)
    result.reverse()
    coloring = greedy_coloring(G, result)
    bound = max(coloring.values()) + 1
    C = [[] for _ in range(bound)]
    for u in G.nodes():
        C[coloring[u]].append(u)
    return {"coloring": C, "lower_bound": bound}



if __name__ == "__main__":

    G = nx.erdos_renyi_graph(50, 0.5, 1)

    print( lb_reverse_sl(G) )
    print( lb_reverse_lf(G) )
    