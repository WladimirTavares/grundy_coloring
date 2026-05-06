"""
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

ILP formulations (exact, solved via OR-Tools / SCIP):

    solver_rodrigues(G, ub)             – partition model [Rod20]
    solver_carvalho(G, ub)              – partition model [Car23]
    solver_carvalho_modificado(G, ub)   – partition + aggregated Grundy cut
    solver_carvalho_representante(G)    – asymmetric representative model [Car23]
    solver_carvalho_representante2(G,ub)– representative model (explicit ub)
    solver_carvalho_representante3(G,order,ub) – representative model with order

----------
.. [Rod20] Rodrigues, E. N. H. D. (2020). Coloração k-imprópria gulosa.
           Repositório UFC. http://www.repositorio.ufc.br/handle/riufc/50955
.. [Car23] Carvalho, M., Melo, R., Santos, M. C., Toso, R. F., and
           Resende, M. G. C. (2023). Formulações de programação inteira para o
           problema da coloração de Grundy. Anais SBPO 2024.
"""

import time
import itertools
from collections import defaultdict, deque
from typing import Callable, Optional
import math
import heapq
import networkx as nx
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from greedy_coloring import is_greedy_coloring
from upper_bound import upper_bound1
from vertex_ordering import smallest_last_ordering



# ---------------------------------------------------------------------------
# ILP formulations
# ---------------------------------------------------------------------------

def get_linear_relaxation(solver):
    """
    Computes the LP relaxation value of a MIP model built with OR-Tools (pywraplp).

    This function temporarily relaxes all integer variables (IntVar, BoolVar)
    to continuous variables, solves the LP, retrieves the objective value,
    and then restores the original integrality.

    Parameters
    ----------
    solver : pywraplp.Solver
        An OR-Tools solver instance (e.g., SCIP) already containing the model.

    Returns
    -------
    float
        Objective value of the linear relaxation.

    Notes
    -----
    - This computes the LP relaxation at the root (not node-level relaxations).
    - The solver state is restored after execution.
    - Assumes the model has already been fully built before calling.
    """

    # guarda quais variáveis eram inteiras
    integer_vars = []
    for var in solver.variables():
        if var.Integer():
            integer_vars.append(var)
            var.SetInteger(False)

    # resolve a relaxação
    status = solver.Solve()

    if status not in (solver.OPTIMAL, solver.FEASIBLE):
        lp_value = None
    else:
        lp_value = solver.Objective().Value()

    # restaura integridade
    for var in integer_vars:
        var.SetInteger(True)

    return lp_value


def solver_rodrigues(
    grafo: nx.Graph,
    upperbound: int,
    time_limit: int = 600000,
) -> dict:
    """Solve the Grundy coloring problem via the Rodrigues partition model.

    Partition-based ILP where binary variables x[v, c] encode the color
    assignment and z[c] indicates whether color c is active.  The Grundy
    property is encoded with **linear** constraints in |C|:

        x[v,c] ≥ 1 − Σ_{d<c} x[v,d] − Σ_{u∈N(v)} x[u,c]   ∀v, c

    This forces x[v,c] = 1 whenever v has no color d < c and no neighbour
    with color c, thereby ensuring the greedy condition.

    Modifications over [Rod20]:

    * Isolated vertices are pinned to color 0.
    * An externally supplied upper bound limits the color palette to C.
    * Symmetry is broken by enforcing z[0] ≥ z[1] ≥ … ≥ z[|C|−1].

    Constraint counts
    -----------------
    * Assignment:      O(|V|)
    * Proper coloring: O(|E| · |C|)
    * Grundy property: O(|V| · |C|)     ← linear in |C|
    * Symmetry:        O(|C|)
    * **Total:**       O(|V| · |C| + |E| · |C|)

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    upperbound : int
        Size of the color palette C = {0, …, upperbound − 1}.
        Typically set to Δ(G) + 1 or the stair factor.
    time_limit : int, optional
        Solver wall-clock time limit in milliseconds (default 600 000 = 10 min).

    Returns
    -------
    dict
        ``{"model", "gamma", "optimal", "cpu_s", "classes", "valid"}``

        * **model** – ``"rodrigues"``.
        * **gamma** – Grundy number found (``None`` if infeasible).
        * **optimal** – ``True`` if the solution is provably optimal.
        * **cpu_s** – solver wall-clock time in seconds.
        * **classes** – Grundy coloring as a list of color-class lists.
        * **valid** – whether *classes* passes :func:`is_greedy_coloring`.

    References
    ----------
    .. [Rod20] Rodrigues (2020), §3.
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)

    # x[v, c] = 1  iff vertex v receives color c
    # z[c]    = 1  iff at least one vertex uses color c (color c is "active")
    x = {(v, c): solver.IntVar(0, 1, f"x[{v},{c}]")
         for v in grafo.nodes for c in range(upperbound)}
    z = {c: solver.IntVar(0, 1, f"z[{c}]") for c in range(upperbound)}

    # (1) Each vertex receives exactly one color.
    for v in grafo.nodes:
        solver.Add(solver.Sum([x[v, c] for c in range(upperbound)]) == 1)

    # (2) z[c] = 1 only if some vertex has color c.
    #     Ensures the objective does not count empty color classes.
    for c in range(upperbound):
        solver.Add(z[c] <= solver.Sum([x[v, c] for v in grafo.nodes]))

    # (3) Adjacent vertices may not share a color class (proper coloring).
    #     x[u,c] + x[v,c] <= z[c] also links x to z: if no vertex uses c,
    #     z[c]=0 and the constraint becomes x[u,c] + x[v,c] <= 0.
    for (u, v) in grafo.edges:
        for c in range(upperbound):
            solver.Add(x[u, c] + x[v, c] <= z[c])

    # (4) Grundy property (linear form): forces x[v,c]=1 when v has no
    #     color d<c and no neighbour with color c.
    #     Derived from: x[v,c] >= 1 - sum_{d<c} x[v,d] - sum_{u in N(v)} x[u,c]
    for v in grafo.nodes:
        for c in range(upperbound):
            solver.Add(
                x[v, c] >= 1
                - solver.Sum(x[v, d] for d in range(c))          # v has no color < c
                - solver.Sum(x[u, c] for u in grafo.adj[v])      # no neighbour has color c
            )

    # (5) Symmetry breaking: colors used in order z[0] >= z[1] >= ...
    #     Eliminates permutation-equivalent solutions.
    for c in range(1, upperbound):
        solver.Add(z[c] <= z[c - 1])

    # (6) Isolated vertices must receive color 0 (smallest available color).
    for v in grafo.nodes:
        if len(grafo.adj[v]) == 0:
            solver.Add(x[v, 0] == 1)

    # Objective: maximize the number of active colors = Gamma(G).
    solver.Maximize(solver.Sum(z[c] for c in range(upperbound)))

    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes = [[] for _ in range(gamma)] if gamma else []
    for v in grafo.nodes:
        for i in range(gamma or 0):
            if x[v, i].solution_value() >= 0.5:
                classes[i].append(v)
                break

    valid = is_greedy_coloring(grafo, classes) if classes else False
    return {
        "model":   "rodrigues",
        "gamma":   gamma,
        "optimal": status == pywraplp.Solver.OPTIMAL,
        "cpu_s":   cpu,
        "classes": classes,
        "valid":   valid,
        "linear_relaxation" : get_linear_relaxation(solver)
    }


def solver_carvalho(
    grafo: nx.Graph,
    upperbound: int,
    time_limit: int = 600000,
) -> dict:
    """Solve the Grundy coloring problem via the Carvalho partition model.

    Identical variable structure to :func:`solver_rodrigues` but expresses
    the Grundy property with **quadratic** per-pair constraints in |C|:

        x[v, c′] ≤ Σ_{u∈N(v)} x[u, c]   ∀v, c < c′

    This formulation allows the SCIP LP relaxation to be tighter for dense
    graphs (fewer fractional solutions pass the pruning test), at the cost of
    O(|V| · |C|²) constraints instead of O(|V| · |C|).

    Constraint counts
    -----------------
    * Assignment:      O(|V|)
    * Proper coloring: O(|E| · |C|)
    * Grundy property: O(|V| · |C|²)    ← quadratic in |C|
    * Symmetry:        O(|C|²)
    * **Total:**       O(|V| · |C|² + |E| · |C|)

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    upperbound : int
        Size of the color palette.
    time_limit : int, optional
        Solver time limit in milliseconds (default 600 000).

    Returns
    -------
    dict
        Same keys as :func:`solver_rodrigues`, with ``"model": "carvalho"``.

    References
    ----------
    .. [Car23] Carvalho et al. (2023), Formulation F2.
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)

    # x[v, c] = 1  iff vertex v receives color c
    # z[c]    = 1  iff color c is used by at least one vertex
    x = {(v, c): solver.IntVar(0, 1, f"x[{v},{c}]")
         for v in grafo.nodes for c in range(upperbound)}
    z = {c: solver.IntVar(0, 1, f"z[{c}]") for c in range(upperbound)}

    # (1) Each vertex receives exactly one color.
    for v in grafo.nodes:
        solver.Add(solver.Sum([x[v, c] for c in range(upperbound)]) == 1)

    # (2) z[c] active only if some vertex uses color c.
    for c in range(upperbound):
        solver.Add(z[c] <= solver.Sum([x[v, c] for v in grafo.nodes]))

    # (3) Adjacent vertices may not share a color class (proper coloring).
    for (u, v) in grafo.edges:
        for c in range(upperbound):
            solver.Add(x[u, c] + x[v, c] <= z[c])

    # (4) Isolated vertices may be assigned any color c, but only if z[c]=1.
    for v in grafo.nodes:
        if len(grafo.adj[v]) == 0:
            for c in range(upperbound):
                solver.Add(x[v, c] <= z[c])

    # (5) Symmetry breaking: z[0] >= z[1] >= ... (quadratic form, all pairs).
    for c in range(upperbound):
        for c_ in range(c + 1, upperbound):
            solver.Add(z[c_] <= z[c])

    # (6) Grundy property (quadratic form): if v has color c' > c, then at
    #     least one neighbour of v must hold color c.
    #     Derived from: x[v,c'] <= sum_{u in N(v)} x[u,c]  for all c < c'
    for v in grafo.nodes:
        for c in range(upperbound):
            for c_ in range(c + 1, upperbound):
                solver.Add(
                    x[v, c_] <= solver.Sum(x[u, c] for u in grafo.adj[v])
                )

    # Objective: maximize the number of active colors = Gamma(G).
    solver.Maximize(solver.Sum(z[c] for c in range(upperbound)))

    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes = [[] for _ in range(gamma)] if gamma else []
    for v in grafo.nodes:
        for i in range(gamma or 0):
            if x[v, i].solution_value() >= 0.5:
                classes[i].append(v)
                break

    valid = is_greedy_coloring(grafo, classes) if classes else False
    return {
        "model":   "carvalho",
        "gamma":   gamma,
        "optimal": status == pywraplp.Solver.OPTIMAL,
        "cpu_s":   cpu,
        "classes": classes,
        "valid":   valid,
        "linear_relaxation" : get_linear_relaxation(solver)
    }


def solver_carvalho_modificado(
    grafo: nx.Graph,
    upperbound: int,
    time_limit: int = 600000,
) -> dict:
    """Solve the Grundy coloring problem via the modified Carvalho partition model.

    Extends :func:`solver_carvalho` by adding an *aggregated Grundy cut*
    (the *Fábio cut*) that sums over all colors above c simultaneously:

        Σ_{c′ > c} x[v, c′] ≤ Σ_{u∈N(v)} x[u, c]   ∀v, c

    This cut is *implied* by the per-pair constraints of
    :func:`solver_carvalho` (set c′ = c+1, c+2, … and sum), but it can
    tighten the LP relaxation when the individual per-pair constraints are
    fractionally satisfied.  In practice, the aggregated cut reduces the
    number of B&B nodes on dense instances.

    Constraint counts
    -----------------
    * Aggregated Grundy cut: O(|V| · |C|)   ← cheaper to enumerate
    * Per-pair Grundy:       O(|V| · |C|²)
    * **Total:**             O(|V| · |C|² + |E| · |C|)  (same order as F2)

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    upperbound : int
        Size of the color palette.
    time_limit : int, optional
        Solver time limit in milliseconds (default 600 000).

    Returns
    -------
    dict
        Same keys as :func:`solver_rodrigues`, with
        ``"model": "carvalho_modificado"``.
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)

    # x[v, c] = 1  iff vertex v receives color c
    # z[c]    = 1  iff color c is used by at least one vertex
    x = {(v, c): solver.IntVar(0, 1, f"x[{v},{c}]")
         for v in grafo.nodes for c in range(upperbound)}
    z = {c: solver.IntVar(0, 1, f"z[{c}]") for c in range(upperbound)}

    # (1) Each vertex receives exactly one color.
    for v in grafo.nodes:
        solver.Add(solver.Sum([x[v, c] for c in range(upperbound)]) == 1)

    # (2) z[c] active only if some vertex uses color c.
    for c in range(upperbound):
        solver.Add(z[c] <= solver.Sum([x[v, c] for v in grafo.nodes]))

    # (3) Adjacent vertices may not share a color class (proper coloring).
    for (u, v) in grafo.edges:
        for c in range(upperbound):
            solver.Add(x[u, c] + x[v, c] <= z[c])

    # (4) Isolated vertices: pinned to color 0.
    for v in grafo.nodes:
        if len(grafo.adj[v]) == 0:
            solver.Add(x[v, 0] <= z[0])

    # (5) Symmetry breaking: z[0] >= z[1] >= ... (linear form).
    for c in range(1, upperbound):
        solver.Add(z[c] <= z[c - 1])

    # (6) Aggregated Grundy cut (Fábio cut): the sum of x[v, c'] for all
    #     c' > c is bounded by the number of neighbours of v with color c.
    #     This is a valid aggregation of the per-pair Grundy constraints and
    #     can tighten the LP relaxation beyond what (7) alone provides.
    for v in grafo.nodes:
        for c in range(upperbound):
            solver.Add(
                solver.Sum(x[v, c_] for c_ in range(c + 1, upperbound))
                <= solver.Sum(x[u, c] for u in grafo.adj[v] if u != v)
            )

    # (7) Per-pair Grundy property (same as solver_carvalho constraint 6):
    #     if v has color c' > c, at least one neighbour must have color c.
    for v in grafo.nodes:
        for c in range(upperbound):
            for c_ in range(c + 1, upperbound):
                solver.Add(
                    x[v, c_] <= solver.Sum(x[u, c] for u in grafo.adj[v])
                )

    # Objective: maximize the number of active colors = Gamma(G).
    solver.Maximize(solver.Sum(z[c] for c in range(upperbound)))

    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes = [[] for _ in range(gamma)] if gamma else []
    for v in grafo.nodes:
        for i in range(gamma or 0):
            if x[v, i].solution_value() >= 0.5:
                classes[i].append(v)
                break

    valid = is_greedy_coloring(grafo, classes) if classes else False
    return {
        "model":   "carvalho_modificado",
        "gamma":   gamma,
        "optimal": status == pywraplp.Solver.OPTIMAL,
        "cpu_s":   cpu,
        "classes": classes,
        "valid":   valid,
        "linear_relaxation" : get_linear_relaxation(solver)
    }


def solver_carvalho_representante1(
    grafo: nx.Graph,
    upperbound: int,
    time_limit: int = 600000,
) -> dict:
    """Asymmetric representative model with an explicit upper-bound parameter.

    Identical to :func:`solver_carvalho_representante` except that the
    potential upper bound for the MTZ constraints is set to *upperbound* (the
    externally supplied color-palette size) rather than n.  This can yield a
    tighter relaxation when upperbound << n.

    Each color class is represented by its lexicographically smallest
    non-adjacent vertex (*representative*).  The ordering of color classes is
    encoded via Miller–Tucker–Zemlin (MTZ) potentials phi[v].

    Variables
    ---------
    x[u, v] : binary
        1 iff vertex v belongs to the color class represented by u
        (u and v must be non-adjacent, u <= v in index order).
    y[u, v] : binary
        1 iff representative u precedes representative v in the coloring order.
    pot[v]  : continuous in [0, upperbound]
        MTZ potential of vertex v; encodes the position of v's class.

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    upperbound : int
        Upper bound on Γ(G), used as the potential ceiling.  Typically
        ``fast_stair_factor(grafo)`` or ``stair_factor(grafo)``.
    time_limit : int, optional
        Solver time limit in milliseconds (default 600 000).

    Returns
    -------
    dict
        Same keys as :func:`solver_rodrigues`.
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)


    nodes = sorted(grafo.nodes())

    # x[u, v] = 1  iff v is in the class represented by u (u <= v, u,v non-adjacent)
    # x[u, u] = 1  iff u is a representative (i.e., u heads a color class)
    x = {}
    for u in nodes:
        for v in nodes:
            if u <= v and v not in grafo.adj[u]:
                x[u, v] = solver.IntVar(0, 1, f"x[{u},{v}]")

    # y[u, v] = 1  iff representative u precedes representative v in ordering
    y = {}
    for v in nodes:
        for u in nodes:
            if u != v:
                y[u, v] = solver.IntVar(0, 1, f"y[{u},{v}]")

    # pot[v]: MTZ potential; pot[u] < pot[v] when y[u,v] = 1
    pot = {v: solver.NumVar(0, upperbound, f"pot[{v}]") for v in nodes}

    # (1) Clique constraint: within a color class (independent set), no two
    #     members v, w of the same class u can be adjacent to each other.
    #     If v and w are both non-adjacent to u but adjacent to each other,
    #     they cannot both belong to u's class.
    for u in grafo.nodes:
        for v in grafo.nodes:
            for w in grafo.nodes:
                if (v not in grafo.adj[u] and w not in grafo.adj[u]
                        and (v, w) in grafo.edges and u <= v and v < w):
                    solver.Add(x[u, v] + x[u, w] <= x[u, u])

    # (2) Coverage: every vertex v belongs to exactly one color class.
    #     The representative u of v's class satisfies u <= v and u ≁ v.
    for u in nodes:
        solver.Add(
            solver.Sum(x[v, u] for v in nodes
                       if v not in grafo.adj[u] and v <= u) == 1
        )

    # (3) Grundy property: if x[u, v] = 1 (v is in class u), then for every
    #     other representative p, either p comes after u (y[p,u]=0), or p's
    #     class has a neighbour of v.  This ensures every color < color(u)
    #     is witnessed by a neighbour of v.
    for u in nodes:
        for p in nodes:
            if p != u:
                for v in nodes:
                    if v not in grafo.adj[u] and u <= v:
                        solver.Add(
                            x[u, v] <= solver.Sum(
                                x[p, w] for w in grafo.nodes
                                if w in grafo.adj[v]
                                and w not in grafo.adj[p]
                                and w >= p
                            ) + 1 - y[p, u]
                        )

    # (4) Membership validity: a vertex v can only be in class u if u is a
    #     representative (x[u,u] = 1).
    for u in nodes:
        for v in nodes:
            if v not in grafo.adj[u] and u < v:
                solver.Add(x[u, v] <= x[u, u])

    # (5) Total ordering lower bound: if both u and v are representatives,
    #     they must be ordered (y[u,v] + y[v,u] >= 1).
    #     Combined with x[u,u] + x[v,v] - 1 >= 0 only when both are active.
    for u in grafo.nodes:
        for v in grafo.nodes:
            if u < v:
                solver.Add(y[v, u] + y[u, v] >= x[u, u] + x[v, v] - 1)

    # (6) Consistency + MTZ ordering with upperbound as big-M:
    #     - y[u,v] + y[v,u] <= x[u,u]: ordering only defined when u is a rep.
    #     - MTZ: if y[u,v]=1 then pot[u] < pot[v]  (pot[u] - pot[v] + 1 <= 0).
    #       When y[u,v]=0 the constraint relaxes to pot[u]-pot[v]+1 <= upperbound,
    #       which is always satisfied given pot ∈ [0, upperbound].
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[v, u] + y[u, v] <= x[u, u])
                solver.Add(pot[u] - pot[v] + 1 <= upperbound * (1 - y[u, v]))

    # Objective: maximize the number of representatives = number of color classes.
    solver.Maximize(solver.Sum(x[v, v] for v in nodes))

    lp_val = get_linear_relaxation(solver)

    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes, rep = [], []
    if feasible:
        for u in grafo.nodes:
            if (u, u) in x and x[u, u].solution_value() > 0.5:
                cor = [u]; rep.append(u)
                for v in grafo.nodes:
                    if v != u and (u, v) in x and x[u, v].solution_value() > 0.5:
                        cor.append(v)
                classes.append(cor)
        # Bubble-sort classes by the y ordering among representatives.
        for _ in range(len(classes)):
            for j in range(len(classes) - 1, 0, -1):
                if y[rep[j - 1], rep[j]].solution_value() < 0.5:
                    rep[j], rep[j - 1]         = rep[j - 1], rep[j]
                    classes[j], classes[j - 1] = classes[j - 1], classes[j]

    valid = is_greedy_coloring(grafo, classes) if classes else False
    lp_gap = (lp_val - gamma) / lp_val if (feasible and gamma and lp_val > 0) else None

    return {
        "model":   "rep fixed order",
        "gamma":   gamma,
        "optimal": status == pywraplp.Solver.OPTIMAL,
        "cpu_s":   cpu,
        "classes": classes,
        "valid":   valid,
        "linear_relaxation" : lp_val,
        "nodes_explored":    solver.nodes(),
        "n_variables":       solver.NumVariables(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }


def solver_carvalho_representante2(
    grafo: nx.Graph,
    order: list[int],
    upperbound: int,
    time_limit: int = 600000,
) -> dict:
    """Representative model with a fixed vertex ordering and explicit upper bound.

    Extends :func:`solver_carvalho_representante1` by:

    * Restricting representative pairs to those consistent with a pre-computed
      vertex ordering *order* (e.g., smallest-last), reducing the number of
      x variables from O(n²) to O(n · α) where α is the degeneracy.
    * Solving the LP relaxation separately with GLOP before the MIP with SCIP,
      providing a tighter initial upper bound and the lp_gap metric.

    The ordering restricts which vertex can be the representative of a class:
    among two non-adjacent vertices u, v, only the one appearing first in
    *order* can represent the other (x[u, v] is created only if pos[u] ≤ pos[v]).

    Variables
    ---------
    x[u, v] : binary
        1 iff v is in the class represented by u, with pos[u] <= pos[v].
    y[u, v] : binary
        1 iff representative u precedes representative v in the Grundy order.
    phi[v]  : continuous in [0, upperbound-1]
        MTZ potential encoding the position of v's class in the Grundy order.

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    order : list[int]
        A permutation of V(G) used to restrict representative assignments.
        Typically produced by :func:`smallest_last_ordering`.
    upperbound : int
        Upper bound on Γ(G), used as big-M in the MTZ constraints and as the
        domain ceiling for phi.
    time_limit : int, optional
        Solver time limit in milliseconds (default 600 000).

    Returns
    -------
    dict
        Keys: ``"model", "gamma", "optimal", "cpu_s", "classes", "valid",
        "linear_relaxation", "nodes_explored", "n_variables",
        "n_constraints", "lp_gap"``.
    """
    nodes  = list(grafo.nodes)
    n      = len(nodes)
    pos    = {v: i for i, v in enumerate(order)}
    adj    = {u: set(grafo.adj[u]) for u in nodes}

    # antiG[u]    : non-neighbours of u (candidates to share u's class)
    # antiGcol[u] : antiG[u] ∪ {u}  (u can also represent itself)
    # anti_set[u] : set version of antiGcol[u] for O(1) membership tests
    antiG    = {u: [v for v in nodes if v != u and v not in adj[u]] for u in nodes}
    antiGcol = {u: antiG[u] + [u] for u in nodes}
    anti_set = {u: set(antiGcol[u]) for u in nodes}

    # ------------------------------------------------------------------ #
    # LP relaxation (GLOP) — identical model with NumVar instead of IntVar #
    # ------------------------------------------------------------------ #
    lp_solver = pywraplp.Solver.CreateSolver('GLOP')

    # x_lp[u, v]: fractional assignment; only created for pos[u] <= pos[v]
    x_lp = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x_lp[u, v] = lp_solver.NumVar(0, 1, f"x[{u},{v}]")

    y_lp  = {(u, v): lp_solver.NumVar(0, 1, f"y[{u},{v}]")
             for u in nodes for v in nodes if u != v}
    phi_lp = {v: lp_solver.NumVar(0, upperbound, f"phi[{v}]") for v in nodes}

    lp_solver.Maximize(lp_solver.Sum(
        x_lp[v, v] for v in nodes if (v, v) in x_lp
    ))

    # (LP-1) Membership validity: v can only be in class u if u is a representative.
    for (u, v) in x_lp:
        if u != v:
            lp_solver.Add(x_lp[u, v] <= x_lp[u, u])

    # (LP-2) Clique cut: within class u, two members v,w that are adjacent
    #         cannot both belong to u (independent set constraint).
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x_lp and (u, w) in x_lp:
                    lp_solver.Add(x_lp[u, v] + x_lp[u, w] <= x_lp[u, u])

    # (LP-3) Coverage: every vertex v is covered by exactly one class.
    for v in nodes:
        cover = [x_lp[u, v] for u in antiG[v] if (u, v) in x_lp]
        cover.append(x_lp[v, v])
        lp_solver.Add(lp_solver.Sum(cover) == 1)

    # (LP-4) Grundy property: if v is in class u and p is another representative
    #         that precedes u (y[p,u]=1), then p's class must contain a neighbour
    #         of v.  This ensures every color < color(u) is covered by v's neighbours.
    for (u, v) in x_lp:
        for p in nodes:
            if p == u:
                continue
            nbrs = lp_solver.Sum(
                x_lp[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x_lp
            )
            lp_solver.Add(x_lp[u, v] <= nbrs + 1 - y_lp[p, u])

    # (LP-5) Total ordering lower bound: both active representatives must be ordered.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[v, u] + y_lp[u, v] >= x_lp[u, u] + x_lp[v, v] - 1)

    # (LP-6) Consistency + MTZ ordering:
    #   - y[u,v] + y[v,u] <= x[u,u]: ordering only active when u is a representative.
    #   - MTZ big-M: phi[u] - phi[v] + 1 <= upperbound*(1 - y[u,v])
    #     forces phi[u] < phi[v] when y[u,v]=1.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[u, v] + y_lp[v, u] <= x_lp[u, u])
                lp_solver.Add(phi_lp[u] - phi_lp[v] + 1 <= upperbound * (1 - y_lp[u, v]))

    lp_solver.Solve()
    lp_val = lp_solver.Objective().Value()

    # ------------------------------------------------------------------ #
    # MIP (SCIP)                                                           #
    # ------------------------------------------------------------------ #
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)

    x = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x[u, v] = solver.IntVar(0, 1, f"x[{u},{v}]")

    y   = {(u, v): solver.IntVar(0, 1, f"y[{u},{v}]")
           for u in nodes for v in nodes if u != v}
    phi = {v: solver.NumVar(0, upperbound-1, f"phi[{v}]") for v in nodes}

    solver.Maximize(solver.Sum(x[v, v] for v in nodes if (v, v) in x))

    # (1) Membership validity.
    for (u, v) in x:
        if u != v:
            solver.Add(x[u, v] <= x[u, u])

    # (2) Clique cut within each class (independent set constraint).
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x and (u, w) in x:
                    solver.Add(x[u, v] + x[u, w] <= x[u, u])

    # (3) Coverage: every vertex belongs to exactly one class.
    for v in nodes:
        cover = [x[u, v] for u in antiG[v] if (u, v) in x]
        cover.append(x[v, v])
        solver.Add(solver.Sum(cover) == 1)

    # (4) Grundy property: every predecessor class covers a neighbour of v.
    for (u, v) in x:
        for p in nodes:
            if p == u:
                continue
            nbrs = solver.Sum(
                x[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x
            )
            solver.Add(x[u, v] <= nbrs + 1 - y[p, u])

    # (5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[v, u] + y[u, v] >= x[u, u] + x[v, v] - 1)

    # (6) Consistency + MTZ big-M ordering.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[u, v] + y[v, u] <= x[u, u])
                solver.Add(phi[u] - phi[v] + 1 <= upperbound * (1 - y[u, v]))

    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes, rep = [], []
    if feasible:
        for u in nodes:
            if (u, u) in x and x[u, u].solution_value() > 0.5:
                cor = [u]; rep.append(u)
                for v in nodes:
                    if v != u and (u, v) in x and x[u, v].solution_value() > 0.5:
                        cor.append(v)
                classes.append(cor)
        # Bubble-sort classes by the y ordering among representatives.
        for _ in range(len(classes)):
            for j in range(len(classes) - 1, 0, -1):
                if y[rep[j - 1], rep[j]].solution_value() < 0.5:
                    rep[j], rep[j - 1]         = rep[j - 1], rep[j]
                    classes[j], classes[j - 1] = classes[j - 1], classes[j]

    valid  = is_greedy_coloring(grafo, classes) if classes else False
    lp_gap = (lp_val - gamma) / lp_val if (feasible and gamma and lp_val > 0) else None

    return {
        "model":             "rep with order",
        "gamma":             gamma,
        "optimal":           status == pywraplp.Solver.OPTIMAL,
        "cpu_s":             cpu,
        "classes":           classes,
        "valid":             valid,
        "linear_relaxation": lp_val,
        "nodes_explored":    solver.nodes(),
        "n_variables":       solver.NumVariables(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }


def solver_carvalho_representante3(
    grafo: nx.Graph,
    order: list[int],
    upperbound: int,
    time_limit: int = 600000,
) -> dict:
    """Representative model with ordering, psi-based bound on y-predecessors.

    Extends :func:`solver_carvalho_representante2` with a combinatorial
    bound on the number of predecessors in the Grundy ordering:

        Σ_{p ≠ v} y[p, v] ≤ K[v] - 1   ∀ v ∈ V

    where K[v] = min(upperbound, ψ(v, Δ(G)+1)) and ψ(v, k) is the
    connected degree sequence value from Shi et al. (2005).  This bound
    directly limits the position of v in the Grundy ordering without
    going through the MTZ potentials phi, making it a strong cut for
    the LP relaxation.

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    order : list[int]
        Vertex ordering (e.g., smallest-last).
    upperbound : int
        Upper bound on Γ(G).
    time_limit : int, optional
        Solver time limit in milliseconds (default 600 000).

    Returns
    -------
    dict
        Same keys as :func:`solver_carvalho_representante2`, with
        ``"model": "rep with order limite y"``.
    """
    nodes  = list(grafo.nodes)
    n      = len(nodes)
    pos    = {v: i for i, v in enumerate(order)}
    adj    = {u: set(grafo.adj[u]) for u in nodes}

    antiG    = {u: [v for v in nodes if v != u and v not in adj[u]] for u in nodes}
    antiGcol = {u: antiG[u] + [u] for u in nodes}
    anti_set = {u: set(antiGcol[u]) for u in nodes}

    # ------------------------------------------------------------------ #
    # LP relaxation (GLOP)                                                 #
    # ------------------------------------------------------------------ #
    lp_solver = pywraplp.Solver.CreateSolver('GLOP')

    x_lp = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x_lp[u, v] = lp_solver.NumVar(0, 1, f"x[{u},{v}]")

    y_lp  = {(u, v): lp_solver.NumVar(0, 1, f"y[{u},{v}]")
             for u in nodes for v in nodes if u != v}
    phi_lp = {v: lp_solver.NumVar(0, upperbound-1, f"phi[{v}]") for v in nodes}

    lp_solver.Maximize(lp_solver.Sum(
        x_lp[v, v] for v in nodes if (v, v) in x_lp
    ))

    # (LP-1) Membership validity.
    for (u, v) in x_lp:
        if u != v:
            lp_solver.Add(x_lp[u, v] <= x_lp[u, u])

    # (LP-2) Clique cut within each class.
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x_lp and (u, w) in x_lp:
                    lp_solver.Add(x_lp[u, v] + x_lp[u, w] <= x_lp[u, u])

    # (LP-3) Coverage: every vertex covered by exactly one class.
    for v in nodes:
        cover = [x_lp[u, v] for u in antiG[v] if (u, v) in x_lp]
        cover.append(x_lp[v, v])
        lp_solver.Add(lp_solver.Sum(cover) == 1)

    # (LP-4) Grundy property.
    for (u, v) in x_lp:
        for p in nodes:
            if p == u:
                continue
            nbrs = lp_solver.Sum(
                x_lp[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x_lp
            )
            lp_solver.Add(x_lp[u, v] <= nbrs + 1 - y_lp[p, u])

    # (LP-5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[v, u] + y_lp[u, v] >= x_lp[u, u] + x_lp[v, v] - 1)

    # (LP-6) Consistency + MTZ big-M.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[u, v] + y_lp[v, u] <= x_lp[u, u])
                lp_solver.Add(phi_lp[u] - phi_lp[v] + 1 <= upperbound * (1 - y_lp[u, v]))

    # Compute psi-based individual bounds K[v] = min(upperbound, psi(v)).
    # K[v] - 1 is the maximum number of classes that can precede v.
    table = psi_table(grafo, delta_1(grafo))
    K = {}
    for v in nodes:
        K[v] = min(upperbound, table[v])

    # (LP-7) psi-based predecessor bound on y: limits position of v directly.
    for v in nodes:
        lp_solver.Add(
            lp_solver.Sum(y_lp[p, v] for p in nodes if p != v) <= K[v]-1
        )

    lp_solver.Solve()
    lp_val = lp_solver.Objective().Value()

    # ------------------------------------------------------------------ #
    # MIP (SCIP)                                                           #
    # ------------------------------------------------------------------ #
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)

    x = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x[u, v] = solver.IntVar(0, 1, f"x[{u},{v}]")

    y   = {(u, v): solver.IntVar(0, 1, f"y[{u},{v}]")
           for u in nodes for v in nodes if u != v}
    phi = {v: solver.NumVar(0, upperbound, f"phi[{v}]") for v in nodes}

    solver.Maximize(solver.Sum(x[v, v] for v in nodes if (v, v) in x))

    # (1) Membership validity.
    for (u, v) in x:
        if u != v:
            solver.Add(x[u, v] <= x[u, u])

    # (2) Clique cut within each class.
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x and (u, w) in x:
                    solver.Add(x[u, v] + x[u, w] <= x[u, u])

    # (3) Coverage.
    for v in nodes:
        cover = [x[u, v] for u in antiG[v] if (u, v) in x]
        cover.append(x[v, v])
        solver.Add(solver.Sum(cover) == 1)

    # (4) Grundy property.
    for (u, v) in x:
        for p in nodes:
            if p == u:
                continue
            nbrs = solver.Sum(
                x[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x
            )
            solver.Add(x[u, v] <= nbrs + 1 - y[p, u])

    # (5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[v, u] + y[u, v] >= x[u, u] + x[v, v] - 1)

    # (6) Consistency + MTZ big-M ordering.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[u, v] + y[v, u] <= x[u, u])
                solver.Add(phi[u] - phi[v] + 1 <= upperbound * (1 - y[u, v]))

    
    table = psi_table(grafo, delta_1(grafo))
    K = {}
    for v in nodes:
        K[v] = min(upperbound, table[v])

    # (7) psi-based predecessor bound on y (same as LP-7).
    for v in nodes:
        solver.Add(
            solver.Sum(y[p, v] for p in nodes if p != v) <= K[v]-1
        )

    lp_val = get_linear_relaxation(solver)


    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes, rep = [], []
    if feasible:
        for u in nodes:
            if (u, u) in x and x[u, u].solution_value() > 0.5:
                cor = [u]; rep.append(u)
                for v in nodes:
                    if v != u and (u, v) in x and x[u, v].solution_value() > 0.5:
                        cor.append(v)
                classes.append(cor)
        for _ in range(len(classes)):
            for j in range(len(classes) - 1, 0, -1):
                if y[rep[j - 1], rep[j]].solution_value() < 0.5:
                    rep[j], rep[j - 1]         = rep[j - 1], rep[j]
                    classes[j], classes[j - 1] = classes[j - 1], classes[j]

    valid  = is_greedy_coloring(grafo, classes) if classes else False
    lp_gap = (lp_val - gamma) / lp_val if (feasible and gamma and lp_val > 0) else None

    return {
        "model":             "rep with ordering, psi-based bound on y-predecessors",
        "gamma":             gamma,
        "optimal":           status == pywraplp.Solver.OPTIMAL,
        "cpu_s":             cpu,
        "classes":           classes,
        "valid":             valid,
        "linear_relaxation": lp_val,
        "nodes_explored":    solver.nodes(),
        "n_variables":       solver.NumVariables(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }


from upper_bound import psi_table, delta_1

def solver_carvalho_representante4(
    grafo: nx.Graph,
    order: list[int],
    U: int,
    time_limit: int = 600000,
) -> dict:
    """Representative model with psi-based phi bound and D&L strengthening.

    Extends :func:`solver_carvalho_representante2` with two tightenings:

    1. **Tighter phi domain**: ``phi[v] <= (U-1) * x[v,v]`` forces phi[v]=0
       for non-representatives, tightening the LP relaxation.

    2. **Desrochers & Laporte (1991) MTZ strengthening**: replaces the
       standard big-M MTZ constraint with:

           phi[u] - phi[v] + 1 <= U*(1 - y[u,v]) - y[v,u]

       This single inequality unifies the original MTZ constraint and its
       strengthening: when y[v,u]=1, the RHS tightens from U to U-1,
       which is valid because y[v,u]=1 implies phi[v] >= 1 (so the
       maximum of phi[u]-phi[v] is U-2, not U-1).

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    order : list[int]
        Vertex ordering (e.g., smallest-last).
    U : int
        Upper bound on Γ(G).
    time_limit : int, optional
        Solver time limit in milliseconds (default 600 000).

    Returns
    -------
    dict
        Same keys as :func:`solver_carvalho_representante2`, with
        ``"model": "rep with order limit phi + fortalecimento de Desrochers & Laporte"``.

    References
    ----------
    .. [DL91] Desrochers, M. and Laporte, G. (1991). Improvements and
              extensions to the Miller-Tucker-Zemlin subtour elimination
              constraints. Operations Research Letters, 10(1):27–36.
    """
    nodes  = list(grafo.nodes)
    n      = len(nodes)
    pos    = {v: i for i, v in enumerate(order)}
    adj    = {u: set(grafo.adj[u]) for u in nodes}

    antiG    = {u: [v for v in nodes if v != u and v not in adj[u]] for u in nodes}
    antiGcol = {u: antiG[u] + [u] for u in nodes}
    anti_set = {u: set(antiGcol[u]) for u in nodes}

    # ------------------------------------------------------------------ #
    # LP relaxation (GLOP)                                                 #
    # ------------------------------------------------------------------ #
    lp_solver = pywraplp.Solver.CreateSolver('GLOP')

    x_lp = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x_lp[u, v] = lp_solver.NumVar(0, 1, f"x[{u},{v}]")

    y_lp  = {(u, v): lp_solver.NumVar(0, 1, f"y[{u},{v}]")
             for u in nodes for v in nodes if u != v}
    phi_lp = {v: lp_solver.NumVar(0, U-1, f"phi[{v}]") for v in nodes}

    lp_solver.Maximize(lp_solver.Sum(
        x_lp[v, v] for v in nodes if (v, v) in x_lp
    ))

    # (LP-1) Membership validity.
    for (u, v) in x_lp:
        if u != v:
            lp_solver.Add(x_lp[u, v] <= x_lp[u, u])

    # (LP-2) Clique cut within each class.
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x_lp and (u, w) in x_lp:
                    lp_solver.Add(x_lp[u, v] + x_lp[u, w] <= x_lp[u, u])

    # (LP-3) Coverage.
    for v in nodes:
        cover = [x_lp[u, v] for u in antiG[v] if (u, v) in x_lp]
        cover.append(x_lp[v, v])
        lp_solver.Add(lp_solver.Sum(cover) == 1)

    # (LP-4) Grundy property.
    for (u, v) in x_lp:
        for p in nodes:
            if p == u:
                continue
            nbrs = lp_solver.Sum(
                x_lp[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x_lp
            )
            lp_solver.Add(x_lp[u, v] <= nbrs + 1 - y_lp[p, u])

    # (LP-5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[v, u] + y_lp[u, v] >= x_lp[u, u] + x_lp[v, v] - 1)

    # Compute per-vertex psi bounds: K[v] = min(U, psi(v))
    table = psi_table(grafo, delta_1(grafo))
    K = {}
    for v in nodes:
        K[v] = min(U, table[v])


    for v in nodes:
        lp_solver.Add(phi_lp[v] <= (K[v]-1)*x_lp[v,v])


    # (LP-7) Consistency + D&L strengthened MTZ:
    #   y[u,v] + y[v,u] <= x[u,u]: ordering only defined for representatives.
    #   D&L: phi[u] - phi[v] + 1 <= U*(1-y[u,v]) - y[v,u]
    #   When y[u,v]=1: RHS=0, forces phi[u] < phi[v].
    #   When y[v,u]=1: RHS=U-1, tighter than plain U (valid since phi[v]>=1).
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[u, v] + y_lp[v, u] <= x_lp[u, u])
                

    lp_solver.Solve()
    lp_val = lp_solver.Objective().Value()

    # ------------------------------------------------------------------ #
    # MIP (SCIP)                                                           #
    # ------------------------------------------------------------------ #
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)

    x = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x[u, v] = solver.IntVar(0, 1, f"x[{u},{v}]")

    y   = {(u, v): solver.IntVar(0, 1, f"y[{u},{v}]")
           for u in nodes for v in nodes if u != v}
    phi = {v: solver.NumVar(0, U-1, f"phi[{v}]") for v in nodes}

    solver.Maximize(solver.Sum(x[v, v] for v in nodes if (v, v) in x))

    # (1) Membership validity.
    for (u, v) in x:
        if u != v:
            solver.Add(x[u, v] <= x[u, u])

    # (2) Clique cut within each class.
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x and (u, w) in x:
                    solver.Add(x[u, v] + x[u, w] <= x[u, u])

    # (3) Coverage.
    for v in nodes:
        cover = [x[u, v] for u in antiG[v] if (u, v) in x]
        cover.append(x[v, v])
        solver.Add(solver.Sum(cover) == 1)

    # (4) Grundy property.
    for (u, v) in x:
        for p in nodes:
            if p == u:
                continue
            nbrs = solver.Sum(
                x[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x
            )
            solver.Add(x[u, v] <= nbrs + 1 - y[p, u])

    # (5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[v, u] + y[u, v] >= x[u, u] + x[v, v] - 1)

    # (6) Consistency + MTZ big-M ordering.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[u, v] + y[v, u] <= x[u, u])
                solver.Add(phi[u] - phi[v] + 1 <= U * (1 - y[u, v]))
    

    # Compute per-vertex psi bounds: K[v] = min(U, psi(v))
    table = psi_table(grafo, delta_1(grafo))
    K = {}
    for v in nodes:
        K[v] = min(U, table[v])


    for v in nodes:
        solver.Add(phi[v] <= (K[v]-1)*x[v,v])

    # (7) psi-based predecessor bound on y (same as LP-7).
    for v in nodes:
        solver.Add(
            solver.Sum(y[p, v] for p in nodes if p != v) <= K[v]-1
        )


            
    lp_val = get_linear_relaxation(solver)


    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes, rep = [], []
    if feasible:
        for u in nodes:
            if (u, u) in x and x[u, u].solution_value() > 0.5:
                cor = [u]; rep.append(u)
                for v in nodes:
                    if v != u and (u, v) in x and x[u, v].solution_value() > 0.5:
                        cor.append(v)
                classes.append(cor)
        for _ in range(len(classes)):
            for j in range(len(classes) - 1, 0, -1):
                if y[rep[j - 1], rep[j]].solution_value() < 0.5:
                    rep[j], rep[j - 1]         = rep[j - 1], rep[j]
                    classes[j], classes[j - 1] = classes[j - 1], classes[j]

    valid  = is_greedy_coloring(grafo, classes) if classes else False
    lp_gap = (lp_val - gamma) / lp_val if (feasible and gamma and lp_val > 0) else None

    return {
        "model":             "rep with order with psi-based phi bound and y",
        "gamma":             gamma,
        "optimal":           status == pywraplp.Solver.OPTIMAL,
        "cpu_s":             cpu,
        "classes":           classes,
        "valid":             valid,
        "linear_relaxation": lp_val,
        "nodes_explored":    solver.nodes(),
        "n_variables":       solver.NumVariables(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }


from upper_bound import psi_table, delta_1

def solver_carvalho_representante5(
    grafo: nx.Graph,
    order: list[int],
    U: int,
    time_limit: int = 600000,
) -> dict:
    """Representative model with psi-based phi and y bounds, plus D&L strengthening.

    Combines all tightenings from :func:`solver_carvalho_representante3` and
    :func:`solver_carvalho_representante4`:

    1. **Tighter phi domain**: ``phi[v] <= (K[v]-1) * x[v,v]``
       where K[v] = min(U, ψ(v)) tightens the potential upper bound
       per-vertex using the connected degree sequence bound ψ.

    2. **psi-based predecessor bound on y**:
       ``Σ_{p≠v} y[p,v] <= K[v] - 1``
       limits the number of classes preceding v without using phi.

    3. **D&L MTZ strengthening**:
       ``phi[u] - phi[v] + 1 <= U*(1 - y[u,v]) - y[v,u]``
       tightens the ordering constraint when y[v,u]=1.

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    order : list[int]
        Vertex ordering (e.g., smallest-last).
    U : int
        Upper bound on Γ(G).
    time_limit : int, optional
        Solver time limit in milliseconds (default 600 000).

    Returns
    -------
    dict
        Same keys as :func:`solver_carvalho_representante2`, with
        ``"model": "rep with order limit phi and y fortalecimento de Desrochers & Laporte"``.
    """
    nodes  = list(grafo.nodes)
    n      = len(nodes)
    pos    = {v: i for i, v in enumerate(order)}
    adj    = {u: set(grafo.adj[u]) for u in nodes}

    antiG    = {u: [v for v in nodes if v != u and v not in adj[u]] for u in nodes}
    antiGcol = {u: antiG[u] + [u] for u in nodes}
    anti_set = {u: set(antiGcol[u]) for u in nodes}

    # ------------------------------------------------------------------ #
    # LP relaxation (GLOP)                                                 #
    # ------------------------------------------------------------------ #
    lp_solver = pywraplp.Solver.CreateSolver('GLOP')

    x_lp = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x_lp[u, v] = lp_solver.NumVar(0, 1, f"x[{u},{v}]")

    y_lp  = {(u, v): lp_solver.NumVar(0, 1, f"y[{u},{v}]")
             for u in nodes for v in nodes if u != v}
    phi_lp = {v: lp_solver.NumVar(0, U-1, f"phi[{v}]") for v in nodes}

    lp_solver.Maximize(lp_solver.Sum(
        x_lp[v, v] for v in nodes if (v, v) in x_lp
    ))

    # (LP-1) Membership validity.
    for (u, v) in x_lp:
        if u != v:
            lp_solver.Add(x_lp[u, v] <= x_lp[u, u])

    # (LP-2) Clique cut within each class.
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x_lp and (u, w) in x_lp:
                    lp_solver.Add(x_lp[u, v] + x_lp[u, w] <= x_lp[u, u])

    # (LP-3) Coverage.
    for v in nodes:
        cover = [x_lp[u, v] for u in antiG[v] if (u, v) in x_lp]
        cover.append(x_lp[v, v])
        lp_solver.Add(lp_solver.Sum(cover) == 1)

    # (LP-4) Grundy property.
    for (u, v) in x_lp:
        for p in nodes:
            if p == u:
                continue
            nbrs = lp_solver.Sum(
                x_lp[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x_lp
            )
            lp_solver.Add(x_lp[u, v] <= nbrs + 1 - y_lp[p, u])

    # (LP-5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[v, u] + y_lp[u, v] >= x_lp[u, u] + x_lp[v, v] - 1)

    
    
    # Compute per-vertex psi bounds: K[v] = min(U, psi(v))
    table = psi_table(grafo, delta_1(grafo))
    K = {}
    for v in nodes:
        K[v] = min(U, table[v])

    # (LP-6) Tighter phi domain using K[v] instead of U.
    for v in nodes:
        lp_solver.Add(phi_lp[v] <= (K[v]-1)*x_lp[v,v])

    # (LP-7) psi-based predecessor bound on y.
    for v in nodes:
        lp_solver.Add(
            lp_solver.Sum(y_lp[p, v] for p in nodes if p != v) <= K[v]-1
        )

    # (LP-8) Consistency + standard MTZ (D&L applied in MIP only here).
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[u, v] + y_lp[v, u] <= x_lp[u, u])
                lp_solver.Add(phi_lp[u] - phi_lp[v] + 1 <= U * (1 - y_lp[u, v]))

    lp_solver.Solve()
    lp_val = lp_solver.Objective().Value()

    # ------------------------------------------------------------------ #
    # MIP (SCIP)                                                           #
    # ------------------------------------------------------------------ #
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)

    x = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x[u, v] = solver.IntVar(0, 1, f"x[{u},{v}]")

    y   = {(u, v): solver.IntVar(0, 1, f"y[{u},{v}]")
           for u in nodes for v in nodes if u != v}
    phi = {v: solver.NumVar(0, U-1, f"phi[{v}]") for v in nodes}

    solver.Maximize(solver.Sum(x[v, v] for v in nodes if (v, v) in x))

    # (1) Membership validity.
    for (u, v) in x:
        if u != v:
            solver.Add(x[u, v] <= x[u, u])

    # (2) Clique cut within each class.
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x and (u, w) in x:
                    solver.Add(x[u, v] + x[u, w] <= x[u, u])

    # (3) Coverage.
    for v in nodes:
        cover = [x[u, v] for u in antiG[v] if (u, v) in x]
        cover.append(x[v, v])
        solver.Add(solver.Sum(cover) == 1)

    # (4) Grundy property.
    for (u, v) in x:
        for p in nodes:
            if p == u:
                continue
            nbrs = solver.Sum(
                x[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x
            )
            solver.Add(x[u, v] <= nbrs + 1 - y[p, u])

    # (5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[v, u] + y[u, v] >= x[u, u] + x[v, v] - 1)
    
    # (6) Consistency + MTZ big-M ordering.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[u, v] + y[v, u] <= x[u, u])
                solver.Add(phi[u] - phi[v] + 1 <= U * (1 - y[u, v]))
    

    # Compute per-vertex psi bounds: K[v] = min(U, psi(v))
    table = psi_table(grafo, U)
    K = {}
    for v in nodes:
        K[v] = min(U, table[v])


    for v in nodes:
        solver.Add(phi[v] <= (K[v]-1)*x[v,v])

    # (7) psi-based predecessor bound on y (same as LP-7).
    for v in nodes:
        solver.Add(
            solver.Sum(y[p, v] for p in nodes if p != v) <= K[v]-1
        )



    lp_val = get_linear_relaxation(solver)

    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes, rep = [], []
    if feasible:
        for u in nodes:
            if (u, u) in x and x[u, u].solution_value() > 0.5:
                cor = [u]; rep.append(u)
                for v in nodes:
                    if v != u and (u, v) in x and x[u, v].solution_value() > 0.5:
                        cor.append(v)
                classes.append(cor)
        for _ in range(len(classes)):
            for j in range(len(classes) - 1, 0, -1):
                if y[rep[j - 1], rep[j]].solution_value() < 0.5:
                    rep[j], rep[j - 1]         = rep[j - 1], rep[j]
                    classes[j], classes[j - 1] = classes[j - 1], classes[j]

    valid  = is_greedy_coloring(grafo, classes) if classes else False
    lp_gap = (lp_val - gamma) / lp_val if (feasible and gamma and lp_val > 0) else None

    return {
        "model":             "rep with ordering with psi-based phi and y bounds, plus D&L strengthening",
        "gamma":             gamma,
        "optimal":           status == pywraplp.Solver.OPTIMAL,
        "cpu_s":             cpu,
        "classes":           classes,
        "valid":             valid,
        "linear_relaxation": lp_val,
        "nodes_explored":    solver.nodes(),
        "n_variables":       solver.NumVariables(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }



def solver_carvalho_representante6(
    grafo: nx.Graph,
    order: list[int],
    upperbound: int,
    time_limit: int = 600000,
) -> dict:
    """Representative model with a fixed vertex ordering and explicit upper bound.

    Extends :func:`solver_carvalho_representante1` by:

    * Restricting representative pairs to those consistent with a pre-computed
      vertex ordering *order* (e.g., smallest-last), reducing the number of
      x variables from O(n²) to O(n · α) where α is the degeneracy.
    * Solving the LP relaxation separately with GLOP before the MIP with SCIP,
      providing a tighter initial upper bound and the lp_gap metric.

    The ordering restricts which vertex can be the representative of a class:
    among two non-adjacent vertices u, v, only the one appearing first in
    *order* can represent the other (x[u, v] is created only if pos[u] ≤ pos[v]).

    Variables
    ---------
    x[u, v] : binary
        1 iff v is in the class represented by u, with pos[u] <= pos[v].
    y[u, v] : binary
        1 iff representative u precedes representative v in the Grundy order.
    phi[v]  : continuous in [0, upperbound-1]
        MTZ potential encoding the position of v's class in the Grundy order.

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    order : list[int]
        A permutation of V(G) used to restrict representative assignments.
        Typically produced by :func:`smallest_last_ordering`.
    upperbound : int
        Upper bound on Γ(G), used as big-M in the MTZ constraints and as the
        domain ceiling for phi.
    time_limit : int, optional
        Solver time limit in milliseconds (default 600 000).

    Returns
    -------
    dict
        Keys: ``"model", "gamma", "optimal", "cpu_s", "classes", "valid",
        "linear_relaxation", "nodes_explored", "n_variables",
        "n_constraints", "lp_gap"``.
    """
    nodes  = list(grafo.nodes)
    n      = len(nodes)
    pos    = {v: i for i, v in enumerate(order)}
    adj    = {u: set(grafo.adj[u]) for u in nodes}

    # antiG[u]    : non-neighbours of u (candidates to share u's class)
    # antiGcol[u] : antiG[u] ∪ {u}  (u can also represent itself)
    # anti_set[u] : set version of antiGcol[u] for O(1) membership tests
    antiG    = {u: [v for v in nodes if v != u and v not in adj[u]] for u in nodes}
    antiGcol = {u: antiG[u] + [u] for u in nodes}
    anti_set = {u: set(antiGcol[u]) for u in nodes}

    # ------------------------------------------------------------------ #
    # LP relaxation (GLOP) — identical model with NumVar instead of IntVar #
    # ------------------------------------------------------------------ #
    lp_solver = pywraplp.Solver.CreateSolver('GLOP')

    # x_lp[u, v]: fractional assignment; only created for pos[u] <= pos[v]
    x_lp = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x_lp[u, v] = lp_solver.NumVar(0, 1, f"x[{u},{v}]")

    y_lp  = {(u, v): lp_solver.NumVar(0, 1, f"y[{u},{v}]")
             for u in nodes for v in nodes if u != v}
    phi_lp = {v: lp_solver.NumVar(0, upperbound, f"phi[{v}]") for v in nodes}

    lp_solver.Maximize(lp_solver.Sum(
        x_lp[v, v] for v in nodes if (v, v) in x_lp
    ))

    # (LP-1) Membership validity: v can only be in class u if u is a representative.
    for (u, v) in x_lp:
        if u != v:
            lp_solver.Add(x_lp[u, v] <= x_lp[u, u])

    # (LP-2) Clique cut: within class u, two members v,w that are adjacent
    #         cannot both belong to u (independent set constraint).
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x_lp and (u, w) in x_lp:
                    lp_solver.Add(x_lp[u, v] + x_lp[u, w] <= x_lp[u, u])

    # (LP-3) Coverage: every vertex v is covered by exactly one class.
    for v in nodes:
        cover = [x_lp[u, v] for u in antiG[v] if (u, v) in x_lp]
        cover.append(x_lp[v, v])
        lp_solver.Add(lp_solver.Sum(cover) == 1)

    # (LP-4) Grundy property: if v is in class u and p is another representative
    #         that precedes u (y[p,u]=1), then p's class must contain a neighbour
    #         of v.  This ensures every color < color(u) is covered by v's neighbours.
    for (u, v) in x_lp:
        for p in nodes:
            if p == u:
                continue
            nbrs = lp_solver.Sum(
                x_lp[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x_lp
            )
            lp_solver.Add(x_lp[u, v] <= nbrs + 1 - y_lp[p, u])

    # (LP-5) Total ordering lower bound: both active representatives must be ordered.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[v, u] + y_lp[u, v] >= x_lp[u, u] + x_lp[v, v] - 1)

    # (LP-6) Consistency + MTZ ordering:
    #   - y[u,v] + y[v,u] <= x[u,u]: ordering only active when u is a representative.
    #   - MTZ big-M: phi[u] - phi[v] + 1 <= upperbound*(1 - y[u,v])
    #     forces phi[u] < phi[v] when y[u,v]=1.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[u, v] + y_lp[v, u] <= x_lp[u, u])
                lp_solver.Add(phi_lp[u] - phi_lp[v] + 1 <= upperbound * (1 - y_lp[u, v]))

    lp_solver.Solve()
    lp_val = lp_solver.Objective().Value()

    # ------------------------------------------------------------------ #
    # MIP (SCIP)                                                           #
    # ------------------------------------------------------------------ #
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)

    x = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x[u, v] = solver.IntVar(0, 1, f"x[{u},{v}]")

    y   = {(u, v): solver.IntVar(0, 1, f"y[{u},{v}]")
           for u in nodes for v in nodes if u != v}
    phi = {v: solver.NumVar(0, upperbound-1, f"phi[{v}]") for v in nodes}

    solver.Maximize(solver.Sum(x[v, v] for v in nodes if (v, v) in x))

    # (1) Membership validity.
    for (u, v) in x:
        if u != v:
            solver.Add(x[u, v] <= x[u, u])

    # (2) Clique cut within each class (independent set constraint).
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x and (u, w) in x:
                    solver.Add(x[u, v] + x[u, w] <= x[u, u])

    # (3) Coverage: every vertex belongs to exactly one class.
    for v in nodes:
        cover = [x[u, v] for u in antiG[v] if (u, v) in x]
        cover.append(x[v, v])
        solver.Add(solver.Sum(cover) == 1)

    # (4) Grundy property: every predecessor class covers a neighbour of v.
    for (u, v) in x:
        for p in nodes:
            if p == u:
                continue
            nbrs = solver.Sum(
                x[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x
            )
            solver.Add(x[u, v] <= nbrs + 1 - y[p, u])

    # (5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[v, u] + y[u, v] >= x[u, u] + x[v, v] - 1)

    # (6) Consistency + MTZ big-M ordering.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[u, v] + y[v, u] <= x[u, u])
                solver.Add(phi[u] - phi[v] + y[u,v] - y[u,v] + (upperbound-1)*(y[u,v]+y[v,u] - 1) <= 0 )

    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes, rep = [], []
    if feasible:
        for u in nodes:
            if (u, u) in x and x[u, u].solution_value() > 0.5:
                cor = [u]; rep.append(u)
                for v in nodes:
                    if v != u and (u, v) in x and x[u, v].solution_value() > 0.5:
                        cor.append(v)
                classes.append(cor)
        # Bubble-sort classes by the y ordering among representatives.
        for _ in range(len(classes)):
            for j in range(len(classes) - 1, 0, -1):
                if y[rep[j - 1], rep[j]].solution_value() < 0.5:
                    rep[j], rep[j - 1]         = rep[j - 1], rep[j]
                    classes[j], classes[j - 1] = classes[j - 1], classes[j]

    valid  = is_greedy_coloring(grafo, classes) if classes else False
    lp_gap = (lp_val - gamma) / lp_val if (feasible and gamma and lp_val > 0) else None

    return {
        "model":             "rep with order",
        "gamma":             gamma,
        "optimal":           status == pywraplp.Solver.OPTIMAL,
        "cpu_s":             cpu,
        "classes":           classes,
        "valid":             valid,
        "linear_relaxation": lp_val,
        "nodes_explored":    solver.nodes(),
        "n_variables":       solver.NumVariables(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }


def solver_carvalho_representante7(
    grafo: nx.Graph,
    order: list[int],
    upperbound: int,
    time_limit: int = 600000,
) -> dict:
    """Representative model with a fixed vertex ordering and explicit upper bound.

    Extends :func:`solver_carvalho_representante1` by:

    * Restricting representative pairs to those consistent with a pre-computed
      vertex ordering *order* (e.g., smallest-last), reducing the number of
      x variables from O(n²) to O(n · α) where α is the degeneracy.
    * Solving the LP relaxation separately with GLOP before the MIP with SCIP,
      providing a tighter initial upper bound and the lp_gap metric.

    The ordering restricts which vertex can be the representative of a class:
    among two non-adjacent vertices u, v, only the one appearing first in
    *order* can represent the other (x[u, v] is created only if pos[u] ≤ pos[v]).

    Variables
    ---------
    x[u, v] : binary
        1 iff v is in the class represented by u, with pos[u] <= pos[v].
    y[u, v] : binary
        1 iff representative u precedes representative v in the Grundy order.
    phi[v]  : continuous in [0, upperbound-1]
        MTZ potential encoding the position of v's class in the Grundy order.

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    order : list[int]
        A permutation of V(G) used to restrict representative assignments.
        Typically produced by :func:`smallest_last_ordering`.
    upperbound : int
        Upper bound on Γ(G), used as big-M in the MTZ constraints and as the
        domain ceiling for phi.
    time_limit : int, optional
        Solver time limit in milliseconds (default 600 000).

    Returns
    -------
    dict
        Keys: ``"model", "gamma", "optimal", "cpu_s", "classes", "valid",
        "linear_relaxation", "nodes_explored", "n_variables",
        "n_constraints", "lp_gap"``.
    """
    nodes  = list(grafo.nodes)
    n      = len(nodes)
    pos    = {v: i for i, v in enumerate(order)}
    adj    = {u: set(grafo.adj[u]) for u in nodes}

    # antiG[u]    : non-neighbours of u (candidates to share u's class)
    # antiGcol[u] : antiG[u] ∪ {u}  (u can also represent itself)
    # anti_set[u] : set version of antiGcol[u] for O(1) membership tests
    antiG    = {u: [v for v in nodes if v != u and v not in adj[u]] for u in nodes}
    antiGcol = {u: antiG[u] + [u] for u in nodes}
    anti_set = {u: set(antiGcol[u]) for u in nodes}

    # ------------------------------------------------------------------ #
    # LP relaxation (GLOP) — identical model with NumVar instead of IntVar #
    # ------------------------------------------------------------------ #
    lp_solver = pywraplp.Solver.CreateSolver('GLOP')

    # x_lp[u, v]: fractional assignment; only created for pos[u] <= pos[v]
    x_lp = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x_lp[u, v] = lp_solver.NumVar(0, 1, f"x[{u},{v}]")

    y_lp  = {(u, v): lp_solver.NumVar(0, 1, f"y[{u},{v}]")
             for u in nodes for v in nodes if u != v}
    phi_lp = {v: lp_solver.NumVar(0, upperbound, f"phi[{v}]") for v in nodes}

    lp_solver.Maximize(lp_solver.Sum(
        x_lp[v, v] for v in nodes if (v, v) in x_lp
    ))

    # (LP-1) Membership validity: v can only be in class u if u is a representative.
    for (u, v) in x_lp:
        if u != v:
            lp_solver.Add(x_lp[u, v] <= x_lp[u, u])

    # (LP-2) Clique cut: within class u, two members v,w that are adjacent
    #         cannot both belong to u (independent set constraint).
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x_lp and (u, w) in x_lp:
                    lp_solver.Add(x_lp[u, v] + x_lp[u, w] <= x_lp[u, u])

    # (LP-3) Coverage: every vertex v is covered by exactly one class.
    for v in nodes:
        cover = [x_lp[u, v] for u in antiG[v] if (u, v) in x_lp]
        cover.append(x_lp[v, v])
        lp_solver.Add(lp_solver.Sum(cover) == 1)

    # (LP-4) Grundy property: if v is in class u and p is another representative
    #         that precedes u (y[p,u]=1), then p's class must contain a neighbour
    #         of v.  This ensures every color < color(u) is covered by v's neighbours.
    for (u, v) in x_lp:
        for p in nodes:
            if p == u:
                continue
            nbrs = lp_solver.Sum(
                x_lp[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x_lp
            )
            lp_solver.Add(x_lp[u, v] <= nbrs + 1 - y_lp[p, u])

    # (LP-5) Total ordering lower bound: both active representatives must be ordered.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[v, u] + y_lp[u, v] >= x_lp[u, u] + x_lp[v, v] - 1)

    # (LP-6) Consistency + MTZ ordering:
    #   - y[u,v] + y[v,u] <= x[u,u]: ordering only active when u is a representative.
    #   - MTZ big-M: phi[u] - phi[v] + 1 <= upperbound*(1 - y[u,v])
    #     forces phi[u] < phi[v] when y[u,v]=1.
    for u in nodes:
        for v in nodes:
            if u != v:
                lp_solver.Add(y_lp[u, v] + y_lp[v, u] <= x_lp[u, u])
                lp_solver.Add(phi_lp[u] - phi_lp[v] + 1 <= upperbound * (1 - y_lp[u, v]))

    lp_solver.Solve()
    lp_val = lp_solver.Objective().Value()

    # ------------------------------------------------------------------ #
    # MIP (SCIP)                                                           #
    # ------------------------------------------------------------------ #
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit)

    x = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x[u, v] = solver.IntVar(0, 1, f"x[{u},{v}]")

    y   = {(u, v): solver.IntVar(0, 1, f"y[{u},{v}]")
           for u in nodes for v in nodes if u != v}
    phi = {v: solver.NumVar(0, upperbound-1, f"phi[{v}]") for v in nodes}

    solver.Maximize(solver.Sum(x[v, v] for v in nodes if (v, v) in x))

    # (1) Membership validity.
    for (u, v) in x:
        if u != v:
            solver.Add(x[u, v] <= x[u, u])

    # (2) Clique cut within each class (independent set constraint).
    for u in nodes:
        pu = pos[u]
        for v in antiG[u]:
            pv = pos[v]
            if pv <= pu:
                continue
            for w in antiG[u]:
                pw = pos[w]
                if pw <= pv:
                    continue
                if w in adj[v] and (u, v) in x and (u, w) in x:
                    solver.Add(x[u, v] + x[u, w] <= x[u, u])

    # (3) Coverage: every vertex belongs to exactly one class.
    for v in nodes:
        cover = [x[u, v] for u in antiG[v] if (u, v) in x]
        cover.append(x[v, v])
        solver.Add(solver.Sum(cover) == 1)

    # (4) Grundy property: every predecessor class covers a neighbour of v.
    for (u, v) in x:
        for p in nodes:
            if p == u:
                continue
            nbrs = solver.Sum(
                x[p, w] for w in adj[v]
                if w in anti_set[p] and (p, w) in x
            )
            solver.Add(x[u, v] <= nbrs + 1 - y[p, u])

    # (5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u < v:
                solver.Add(y[v, u] + y[u, v] >= x[u, u] + x[v, v] - 1)

    # (6) Consistency + MTZ big-M ordering.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[u, v] + y[v, u] <= x[u, u])
                solver.Add(phi[u] - phi[v] + y[u,v] - y[u,v] + (upperbound-1)*(y[u,v]+y[v,u] - 1) <= 0 )

    # Compute per-vertex psi bounds: K[v] = min(U, psi(v))
    table = psi_table(grafo, delta_1(grafo))
    K = {}
    for v in nodes:
        K[v] = min(upperbound, table[v])


    for v in nodes:
        solver.Add(phi[v] <= (K[v]-1)*x[v,v])

    # (7) psi-based predecessor bound on y (same as LP-7).
    for v in nodes:
        solver.Add(
            solver.Sum(y[p, v] for p in nodes if p != v) <= K[v]-1
        )


    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes, rep = [], []
    if feasible:
        for u in nodes:
            if (u, u) in x and x[u, u].solution_value() > 0.5:
                cor = [u]; rep.append(u)
                for v in nodes:
                    if v != u and (u, v) in x and x[u, v].solution_value() > 0.5:
                        cor.append(v)
                classes.append(cor)
        # Bubble-sort classes by the y ordering among representatives.
        for _ in range(len(classes)):
            for j in range(len(classes) - 1, 0, -1):
                if y[rep[j - 1], rep[j]].solution_value() < 0.5:
                    rep[j], rep[j - 1]         = rep[j - 1], rep[j]
                    classes[j], classes[j - 1] = classes[j - 1], classes[j]

    valid  = is_greedy_coloring(grafo, classes) if classes else False
    lp_gap = (lp_val - gamma) / lp_val if (feasible and gamma and lp_val > 0) else None

    return {
        "model":             "rep with order",
        "gamma":             gamma,
        "optimal":           status == pywraplp.Solver.OPTIMAL,
        "cpu_s":             cpu,
        "classes":           classes,
        "valid":             valid,
        "linear_relaxation": lp_val,
        "nodes_explored":    solver.nodes(),
        "n_variables":       solver.NumVariables(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }



def repr_formulation(
    grafo: nx.Graph,
    vertex_ordering : bool = True,
    phi_bound_constraint : bool = True,
    predecessor_bound_constraint : bool = True,
    time_limit: int = 600,
) -> dict:
    """
    Variables
    ---------
    x[u, v] : binary
        1 iff v is in the class represented by u, with pos[u] <= pos[v].
    y[u, v] : binary
        1 iff representative u precedes representative v in the Grundy order.
    phi[v]  : continuous in [0, upperbound-1]
        MTZ potential encoding the position of v's class in the Grundy order.

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    order : list[int]
        A permutation of V(G) used to restrict representative assignments.
        Typically produced by :func:`smallest_last_ordering`.
    upperbound : int
        Upper bound on Γ(G), used as big-M in the MTZ constraints and as the
        domain ceiling for phi.
    time_limit : int, optional
        Solver time limit in seconds.

    Returns
    -------
    dict
        Keys: ``"model", "gamma", "optimal", "cpu_s", "classes", "valid",
        "linear_relaxation", "nodes_explored", "n_variables",
        "n_constraints", "lp_gap"``.
    """
    nodes  = list(grafo.nodes)
    n      = len(nodes)

    if vertex_ordering:
        order = smallest_last_ordering(grafo)
    else:
        order = sorted(grafo.nodes())

    upperbound = upper_bound1(grafo)



    pos    = {v: i for i, v in enumerate(order)}
    adj    = {u: set(grafo.adj[u]) for u in nodes}

    # antiG[u]    : non-neighbours of u (candidates to share u's class)
    # antiGcol[u] : antiG[u] ∪ {u}  (u can also represent itself)
    # anti_set[u] : set version of antiGcol[u] for O(1) membership tests
    antiG    = {u: [v for v in nodes if v != u and v not in adj[u]] for u in nodes}
    antiGcol = {u: antiG[u] + [u] for u in nodes}
    anti_set = {u: set(antiGcol[u]) for u in nodes}

    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit*1000)

    x = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x[u, v] = solver.IntVar(0, 1, f"x[{u},{v}]")

    y   = {(u, v): solver.IntVar(0, 1, f"y[{u},{v}]")
           for u in nodes for v in nodes if u != v}
    phi = {v: solver.NumVar(0, upperbound-1, f"phi[{v}]") for v in nodes}

    solver.Maximize(solver.Sum(x[v, v] for v in nodes if (v, v) in x))

    # (1) Membership validity.
    for (u, v) in x:
        if u != v:
            solver.Add(x[u, v] <= x[u, u])

    # (2) Clique cut within each class (independent set constraint).
    for u in grafo.nodes:
        for v in grafo.nodes:
            for w in grafo.nodes:
                if (v not in grafo.adj[u] and w not in grafo.adj[u]
                        and (v, w) in grafo.edges and pos[u] <= pos[v] and pos[v] < pos[w]):
                    solver.Add(x[u, v] + x[u, w] <= x[u, u])



    # (3) Coverage: every vertex belongs to exactly one class.
    for v in nodes:
        cover = [x[u, v] for u in antiG[v] if (u, v) in x]
        cover.append(x[v, v])
        solver.Add(solver.Sum(cover) == 1)

    # (4) Grundy property: every predecessor class covers a neighbour of v.
    for u in nodes:
        for p in nodes:
            if pos[p] != pos[u]:
                for v in nodes:
                    if v not in grafo.adj[u] and pos[u] <= pos[v]:
                        solver.Add(
                            x[u, v] <= solver.Sum(
                                x[p, w] for w in grafo.nodes
                                if w in grafo.adj[v]
                                and w not in grafo.adj[p]
                                and pos[w] >= pos[p]
                            ) + 1 - y[p, u]
                        )


    # (5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u < v:
                solver.Add(y[v, u] + y[u, v] >= x[u, u] + x[v, v] - 1)

    # (6) Consistency + MTZ big-M ordering.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[u, v] + y[v, u] <= x[u, u])
                solver.Add(phi[u] - phi[v] + 1 <= upperbound * (1 - y[u, v]))
    
                
    # Compute per-vertex psi bounds: K[v] = min(U, psi(v))
    table = psi_table(grafo, delta_1(grafo))
    K = {}
    for v in nodes:
        K[v] = min(upperbound, table[v])


    if phi_bound_constraint:
        for v in nodes:
            solver.Add(phi[v] <= (K[v]-1)*x[v,v])

    if predecessor_bound_constraint:
        for v in nodes:
            solver.Add(
                solver.Sum(y[p, v] for p in nodes if p != v) <= K[v]-1
            )

    lp_val = get_linear_relaxation(solver)
    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes, rep = [], []
    if feasible:
        for u in nodes:
            if (u, u) in x and x[u, u].solution_value() > 0.5:
                cor = [u]; rep.append(u)
                for v in nodes:
                    if v != u and (u, v) in x and x[u, v].solution_value() > 0.5:
                        cor.append(v)
                classes.append(cor)
        # Bubble-sort classes by the y ordering among representatives.
        for _ in range(len(classes)):
            for j in range(len(classes) - 1, 0, -1):
                if y[rep[j - 1], rep[j]].solution_value() < 0.5:
                    rep[j], rep[j - 1]         = rep[j - 1], rep[j]
                    classes[j], classes[j - 1] = classes[j - 1], classes[j]

    valid  = is_greedy_coloring(grafo, classes) if classes else False
    lp_gap = (lp_val - gamma) / lp_val if (feasible and gamma and lp_val > 0) else None

    return {
        "model":             "representant",
        "gamma":             gamma,
        "optimal":           status == pywraplp.Solver.OPTIMAL,
        "cpu_s":             cpu,
        "classes":           classes,
        "valid":             valid,
        "linear_relaxation": lp_val,
        "nodes_explored":    solver.nodes(),
        "n_variables":       solver.NumVariables(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }


from lower_bound import lb_reverse_lf, lb_reverse_sl
from vertex_ordering import large_clique


def repr_formulation2(
    grafo: nx.Graph,
    vertex_ordering : bool = True,
    phi_bound_constraint : bool = True,
    predecessor_bound_constraint : bool = True,
    sum_phi_values_constraint : bool = True,
    mtz_strengthening : bool = True,
    time_limit: int = 600,
) -> dict:
    """
    Variables
    ---------
    x[u, v] : binary
        1 iff v is in the class represented by u, with pos[u] <= pos[v].
    y[u, v] : binary
        1 iff representative u precedes representative v in the Grundy order.
    phi[v]  : continuous in [0, upperbound-1]
        MTZ potential encoding the position of v's class in the Grundy order.

    Parameters
    ----------
    grafo : nx.Graph
        Input graph.
    order : list[int]
        A permutation of V(G) used to restrict representative assignments.
        Typically produced by :func:`smallest_last_ordering`.
    upperbound : int
        Upper bound on Γ(G), used as big-M in the MTZ constraints and as the
        domain ceiling for phi.
    time_limit : int, optional
        Solver time limit in seconds.

    Returns
    -------
    dict
        Keys: ``"model", "gamma", "optimal", "cpu_s", "classes", "valid",
        "linear_relaxation", "nodes_explored", "n_variables",
        "n_constraints", "lp_gap"``.
    """
    nodes  = list(grafo.nodes)
    n      = len(nodes)

    if vertex_ordering:
        order = smallest_last_ordering(grafo)
    else:
        order = sorted(grafo.nodes())

    upperbound = upper_bound1(grafo)



    pos    = {v: i for i, v in enumerate(order)}
    adj    = {u: set(grafo.adj[u]) for u in nodes}

    # antiG[u]    : non-neighbours of u (candidates to share u's class)
    # antiGcol[u] : antiG[u] ∪ {u}  (u can also represent itself)
    # anti_set[u] : set version of antiGcol[u] for O(1) membership tests
    antiG    = {u: [v for v in nodes if v != u and v not in adj[u]] for u in nodes}
    antiGcol = {u: antiG[u] + [u] for u in nodes}
    anti_set = {u: set(antiGcol[u]) for u in nodes}

    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.set_time_limit(time_limit*1000)

    x = {}
    for i, u in enumerate(order):
        for j in range(i, len(order)):
            v = order[j]
            if v == u or v in antiG[u]:
                x[u, v] = solver.IntVar(0, 1, f"x[{u},{v}]")

    y   = {(u, v): solver.IntVar(0, 1, f"y[{u},{v}]")
           for u in nodes for v in nodes if u != v}
    phi = {v: solver.NumVar(0, upperbound-1, f"phi[{v}]") for v in nodes}

    solver.Maximize(solver.Sum(x[v, v] for v in nodes if (v, v) in x))

    # (1) Membership validity.
    for (u, v) in x:
        if u != v:
            solver.Add(x[u, v] <= x[u, u])

    # (2) Clique cut within each class (independent set constraint).
    for u in grafo.nodes:
        for v in grafo.nodes:
            for w in grafo.nodes:
                if (v not in grafo.adj[u] and w not in grafo.adj[u]
                        and (v, w) in grafo.edges and pos[u] <= pos[v] and pos[v] < pos[w]):
                    solver.Add(x[u, v] + x[u, w] <= x[u, u])



    # (3) Coverage: every vertex belongs to exactly one class.
    for v in nodes:
        cover = [x[u, v] for u in antiG[v] if (u, v) in x]
        cover.append(x[v, v])
        solver.Add(solver.Sum(cover) == 1)

    # (4) Grundy property: every predecessor class covers a neighbour of v.
    for u in nodes:
        for p in nodes:
            if pos[p] != pos[u]:
                for v in nodes:
                    if v not in grafo.adj[u] and pos[u] <= pos[v]:
                        solver.Add(
                            x[u, v] <= solver.Sum(
                                x[p, w] for w in grafo.nodes
                                if w in grafo.adj[v]
                                and w not in grafo.adj[p]
                                and pos[w] >= pos[p]
                            ) + 1 - y[p, u]
                        )


    # (5) Total ordering lower bound.
    for u in nodes:
        for v in nodes:
            if u < v:
                solver.Add(y[v, u] + y[u, v] >= x[u, u] + x[v, v] - 1)

    # Compute per-vertex psi bounds: K[v] = min(U, psi(v))
    table = psi_table(grafo, delta_1(grafo))
    K = {}
    for v in nodes:
        K[v] = min(upperbound, table[v])


    # (6) Consistency + MTZ big-M ordering.
    for u in nodes:
        for v in nodes:
            if u != v:
                solver.Add(y[u, v] + y[v, u] <= x[u, u])
                
                if mtz_strengthening:
                    solver.Add(phi[u] - phi[v] + 1 <= K[u] * (1 - y[u, v]))
                else:
                    solver.Add(phi[u] - phi[v] + 1 <= upperbound * (1 - y[u, v]))
    

    if phi_bound_constraint:
        for v in nodes:
            solver.Add(phi[v] <= (K[v]-1)*x[v,v])

    if predecessor_bound_constraint:
        for v in nodes:
            solver.Add(
                solver.Sum(y[p, v] for p in nodes if p != v) <= K[v]-1
            )

    lb1 = lb_reverse_sl(grafo)["lower_bound"]
    lb2 = lb_reverse_lf(grafo)["lower_bound"]

    lb = lb1
    if lb2 > lb:
        lb = lb2

    sum_labels = (lb*(lb-1))//2
    
    if sum_phi_values_constraint:
        solver.Add( solver.Sum (phi[v] for v in nodes) >= sum_labels)    



    lp_val = get_linear_relaxation(solver)
    start  = time.time()
    status = solver.Solve()
    cpu    = time.time() - start

    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    gamma    = int(round(solver.Objective().Value())) if feasible else None

    classes, rep = [], []
    if feasible:
        for u in nodes:
            if (u, u) in x and x[u, u].solution_value() > 0.5:
                cor = [u]; rep.append(u)
                for v in nodes:
                    if v != u and (u, v) in x and x[u, v].solution_value() > 0.5:
                        cor.append(v)
                classes.append(cor)
        # Bubble-sort classes by the y ordering among representatives.
        for _ in range(len(classes)):
            for j in range(len(classes) - 1, 0, -1):
                if y[rep[j - 1], rep[j]].solution_value() < 0.5:
                    rep[j], rep[j - 1]         = rep[j - 1], rep[j]
                    classes[j], classes[j - 1] = classes[j - 1], classes[j]

    valid  = is_greedy_coloring(grafo, classes) if classes else False
    lp_gap = (lp_val - gamma) / lp_val if (feasible and gamma and lp_val > 0) else None

    return {
        "model":             "representant",
        "gamma":             gamma,
        "optimal":           status == pywraplp.Solver.OPTIMAL,
        "cpu_s":             cpu,
        "classes":           classes,
        "valid":             valid,
        "linear_relaxation": lp_val,
        "nodes_explored":    solver.nodes(),
        "n_variables":       solver.NumVariables(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }






def run_performance_tests() -> str:
    """
    Executa testes de desempenho sobre um conjunto de solvers para o número
    de Grundy e produz uma tabela LaTeX comparativa.

    Para cada combinação de grafo de Erdős–Rényi G(n, p) e solver, mede
    tempo de CPU, acertos/faltas de cache e pico de memória, e formata os
    resultados em uma tabela LaTeX pronta para inclusão em documentos com
    o pacote ``booktabs``.

    NOTA SOBRE O PICO DE MEMÓRIA
    ----------------------------
    O pico reportado (coluna "peak") é extraído diretamente do campo
    ``"memoria_pico"`` retornado pelo solver, que deve medi-lo
    internamente via ``tracemalloc`` de forma isolada. Esta função não
    reexecuta uma medição externa de memória porque ``tracemalloc``
    capturaria alocações do próprio benchmark (estruturas do networkx,
    frames Python, etc.), contaminando o resultado. Caso o solver não
    forneça ``"memoria_pico"``, o valor reportado é 0.

    Parameters
    ----------
    solvers : list of tuple (name, callable, param)
        Cada elemento é uma tripla:
        - name   (str)      : rótulo do solver exibido no cabeçalho LaTeX.
        - solver (callable) : função com assinatura ``solver(G, param) -> dict``.
          O dict retornado deve conter as chaves:
            "gamma"        (int)   : número de Grundy calculado.
            "hits"         (int)   : acertos de cache (0 se não aplicável).
            "misses"       (int)   : faltas de cache (0 se não aplicável).
            "memoria_pico" (int)   : pico de memória em bytes, medido
                                     internamente pelo solver.
            "valid"        (bool)  : True se a coloração é válida (opcional,
                                     usado apenas para consistência cruzada).
        - param  (any)      : segundo argumento repassado ao solver
                              (ex.: maxsize do lru_cache).

    Returns
    -------
    str
        String LaTeX contendo um ambiente ``tabular`` envolto em
        ``\\resizebox{\\textwidth}{!}{...}``, compatível com ``booktabs``.
        Cada linha corresponde a um grafo; colunas por solver mostram
        tempo (s), hits, misses e pico (KB). A coluna "Val" exibe
        ``\\checkmark`` se todos os solvers concordam no valor de Γ,
        ou ``\\times`` caso contrário.

    Notes
    -----
    Grafos de teste: Erdős–Rényi G(n, p) com n ∈ {10, 15, 20} e
    p ∈ {0.1, 0.2, ..., 0.9}, todos com seed determinística (42 + i,
    onde i é o índice de p), totalizando 27 instâncias.

    O cache do solver é limpo entre execuções (via ``cache_clear()``)
    quando o atributo existir, garantindo que hits/misses reflitam
    apenas a instância corrente.

    O tempo de CPU é medido com ``time.perf_counter()`` em torno da
    chamada ao solver, sobrescrevendo o campo ``"cpu_s"`` eventualmente
    retornado pelo solver (que pode incluir overhead interno de
    ``tracemalloc``).

    Examples
    --------
    >>> solvers = [("MyAlgo", grundy_bitmask, None)]
    >>> latex = run_performance_tests(solvers)
    >>> print(latex[:80])
    \\resizebox{\\textwidth}{!}{
    """
    import time
    import networkx as nx

    sizes = [10, 15, 20, 25, 30]
    #sizes = [10, 15]
    
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    perf = [
        (f"G({n},{p})", nx.erdos_renyi_graph(n, p, seed=42 + i))
        for n in sizes
        for i, p in enumerate(probs)
    ]

    model_names = [name for name, solver, ub_func, ord_func in solvers]

    latex = []
    latex.append("\\resizebox{\\textwidth}{!}{")
    #latex.append("\\begin{tabular}{l c " + " ".join(["r | r |r r"] * len(model_names)) + " c}")
    latex.append("\\begin{tabular}{l | " + " | ".join(["r r r"] * len(model_names)) + " | c}")
    latex.append("\\toprule")

    # Linha 1: nomes dos modelos (multicolumn)
    header1 = ["Graph"]
    for m in model_names:
        header1.append(f"\\multicolumn{{3}}{{c}}{{{m}}}")
    header1.append("Val")

    latex.append(" & ".join(header1) + " \\\\")

    # Linha 2: subcolunas
    header2 = [""]
    for _ in model_names:
        header2 += ["$\\Gamma$", "time", "lp"]
    header2.append("")

    latex.append(" & ".join(header2) + " \\\\")
    latex.append("\\midrule")

    for name, G in perf:
        row_results = []

        for _, solver, ub_fun, ord_fun in solvers:
            if ord_fun is None:
                r = solver(G, ub_fun(G), 1800000)
            else:
                r = solver(G, ord_fun(G), ub_fun(G), 1800000)

            r = dict(r)
            row_results.append(r)

        gamma = row_results[0]["gamma"]
        valid = all(r["gamma"] == gamma for r in row_results)

        min_cpu = min(r["cpu_s"] for r in row_results)
        min_lp  = min(r["linear_relaxation"] for r in row_results)

        row = [name]

        for r in row_results:
            gamma = f"{r['gamma']}"
            cpu = f"{r['cpu_s']:.4f}"
            lp = f"{r['linear_relaxation']:.4g}"


            if r["cpu_s"] == min_cpu:
                cpu = f"\\textbf{{{cpu}}}"

            if r["linear_relaxation"] == min_lp:
                lp = f"\\textbf{{{lp}}}"

            row += [gamma, cpu, lp]

        row.append("\\checkmark" if valid else "\\times")
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    return "\n".join(latex)

def _crown(n):
    """K_{n,n} minus a perfect matching. Γ = n."""
    G = nx.complete_bipartite_graph(n, n)
    G.remove_edges_from([(i, n + i) for i in range(n)])
    return nx.convert_node_labels_to_integers(G)

def _friendship(k):
    """k triangles sharing a single central vertex."""
    G = nx.Graph()
    node = 1
    for _ in range(k):
        a, b = node, node + 1
        G.add_edges_from([(0, a), (0, b), (a, b)])
        node += 2
    return G

def _gem():
    """P_4 + one vertex adjacent to all path vertices. Γ=4."""
    return nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4),
                        (4, 0), (1, 3), (1, 4)])

def _kite():
    """K_4 minus one edge, plus a pendant. Γ=4."""
    return nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4)])

def _house():
    """C_4 with a triangular 'roof'. Γ=4."""
    return nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0),
                        (0, 2), (2, 4), (3, 4)])


def run_correctness_tests() -> list:
    """Test a list of solvers on instances with known Grundy numbers.

    Runs each solver on a fixed battery of 19 graphs (paths, cycles, complete
    graphs, bipartite graphs, and Erdős–Rényi random graphs) and prints a
    formatted table comparing the results.

    Parameters
    ----------
    solvers : list of (str, callable, callable, callable | None)
        Each entry is a 4-tuple ``(name, solver_fn, ub_fn, order_fn)``:

        * **name** – display name for the column header.
        * **solver_fn** – the solver function to call.
        * **ub_fn** – upper-bound function passed to the solver.
        * **order_fn** – if not ``None``, called on G to produce a vertex
          ordering passed as the first positional argument to *solver_fn*.

    Returns
    -------
    list
        Raw results list of ``(name, G, chi_exp, results)`` for further
        inspection or post-processing.

    Notes
    -----
    A test is marked **OK** (✓) if all solvers agree on the Grundy number,
    the number matches the expected value (when known), and all returned
    colorings pass :func:`is_greedy_coloring`.
    
    References
    ----------
    The expected Grundy values come from the following sources:
 
    * **Complete graphs K_n**: Γ(K_n) = n (trivial).
    * **Paths P_n**: Γ(P_n) = 3 for n ≥ 4, Γ(P_2)=2, Γ(P_3)=2.
      [Christen & Selkow, 1979]
    * **Cycles C_n**: Γ(C_n) = 2 if n=4, else 3 for all n ≥ 3.
      [Folklore; verified by brute force]
    * **Complete bipartite K_{m,n}**: Γ(K_{m,n}) = 2 for all m,n ≥ 1.
      [Wikipedia: "complete bipartite graphs are the only connected
       graphs whose Grundy number is two"]
    * **Crown graphs S_n^0 (K_{n,n} minus matching)**: Γ(crown(n)) = n.
      [Wikipedia: Grundy number article, crown graph section]
    * **Stars K_{1,n}**: Γ(K_{1,n}) = 2 for all n ≥ 1.
      [Immediate: one vertex adjacent to all others; max 2 colors]
    * **Hypercubes Q_k**: Γ(Q_1)=2, Γ(Q_2)=2, Γ(Q_3)=4, Γ(Q_4)=5.
      [Hoffman & Johnson Jr. (1999); verified brute-force for Q_3]
    * **Grid graphs P_m □ P_n**: Γ(2×2)=2, Γ(2×3)=4, Γ(3×3)=4.
      [Effantin & Kheddouci (2007), n-dimensional meshes result]
    * **Wheels W_n = K_1 + C_{n-1}**: Γ(W_5)=3, Γ(W_6..W_9)=4.
      [Brute-force verified]
    * **Ladder graphs L_n (P_2 □ P_n)**: Γ(L_4)=4.
      [Brute-force verified]
    * **Petersen graph**: Γ=4.
      [Computational, standard benchmark; Silva et al. 2024]
    * **Mycielski graphs**: Γ(M_3)=3, Γ(M_4)=5.
      [Computational benchmark; Silva et al. 2024]
    * **Empty graph n vertices**: Γ=1 (all vertices get color 1).
    * **Friendship/bowtie F_k (k triangles sharing a vertex)**:
      Γ(F_2)=3, Γ(F_3)=3. [Brute-force verified]
    * **Paw (K_3 + pendant)**: Γ=3. [Brute-force verified]
    * **Diamond (K_4 - edge)**: Γ=3. [Brute-force verified]
    * **Gem (P_4 + universal vertex)**: Γ=4. [Brute-force verified]
    * **Kite**: Γ=4. [Brute-force verified]
    * **House**: Γ=4. [Brute-force verified]
    * **Lollipop Lol(3,2)**: Γ=3. [Brute-force verified]
    * **Bull graph**: Γ=3. [Brute-force verified]
    * **Chvátal graph**: Γ=5. [Sampling-verified; 12-vertex 4-regular]

    """
    
 
 
    tests = [
        # ── Trivial / degenerate ─────────────────────────────────────────────
        ("Empty n=1",        nx.empty_graph(1),                    1),
        ("Empty n=5",        nx.empty_graph(5),                    1),
        ("K_2 (single edge)",nx.complete_graph(2),                 2),
 
        # ── Complete graphs Γ(K_n)=n ─────────────────────────────────────────
        ("Complete K_3",     nx.complete_graph(3),                 3),
        ("Complete K_4",     nx.complete_graph(4),                 4),
        ("Complete K_5",     nx.complete_graph(5),                 5),
        ("Complete K_6",     nx.complete_graph(6),                 6),
 
        # ── Paths Γ(P_n)=3 for n≥4 ───────────────────────────────────────────
        ("Path P_2",         nx.path_graph(2),                     2),
        ("Path P_3",         nx.path_graph(3),                     2),
        ("Path P_4",         nx.path_graph(4),                     3),
        ("Path P_5",         nx.path_graph(5),                     3),
        ("Path P_6",         nx.path_graph(6),                     3),
        ("Path P_7",         nx.path_graph(7),                     3),
        ("Path P_10",        nx.path_graph(10),                    3),
 
        # ── Cycles: Γ=2 only for C_4, else 3 ────────────────────────────────
        ("Cycle C_3 (=K_3)", nx.cycle_graph(3),                    3),
        ("Cycle C_4",        nx.cycle_graph(4),                    2),
        ("Cycle C_5",        nx.cycle_graph(5),                    3),
        ("Cycle C_6",        nx.cycle_graph(6),                    3),
        ("Cycle C_7",        nx.cycle_graph(7),                    3),
        ("Cycle C_8",        nx.cycle_graph(8),                    3),
        ("Cycle C_11",       nx.cycle_graph(11),                   3),
        #("Cycle C_20",       nx.cycle_graph(20),                   3),
 
        # ── Stars Γ=2 for all K_{1,n} ────────────────────────────────────────
        ("Star K_{1,3}",     nx.star_graph(3),                     2),
        ("Star K_{1,4}",     nx.star_graph(4),                     2),
        ("Star K_{1,7}",     nx.star_graph(7),                     2),
 
        # ── Complete bipartite Γ=2 ───────────────────────────────────────────
        ("Bipartite K_{2,3}",nx.complete_bipartite_graph(2, 3),    2),
        ("Bipartite K_{3,3}",nx.complete_bipartite_graph(3, 3),    2),
        ("Bipartite K_{4,4}",nx.complete_bipartite_graph(4, 4),    2),
        #("Bipartite K_{5,5}",nx.complete_bipartite_graph(5, 5),    2),
 
        # ── Crown graphs Γ(crown(n))=n ───────────────────────────────────────
        ("Crown S_2",        _crown(2),                            2),
        ("Crown S_3",        _crown(3),                            3),
        ("Crown S_4",        _crown(4),                            4),
        ("Crown S_5",        _crown(5),                            5),
 
        # ── Hypercubes ───────────────────────────────────────────────────────
        ("Hypercube Q_2",    nx.hypercube_graph(2),                2),
        ("Hypercube Q_3",    nx.hypercube_graph(3),                4),
        ("Hypercube Q_4",    nx.hypercube_graph(4),                5),
 
        # ── Grid graphs (2D meshes) ───────────────────────────────────────────
        ("Grid 2x2",         nx.grid_2d_graph(2, 2),              2),
        ("Grid 2x3",         nx.grid_2d_graph(2, 3),              4),
        ("Grid 3x3",         nx.grid_2d_graph(3, 3),              4),
 
        # ── Ladder graphs (P_2 □ P_n) ────────────────────────────────────────
        ("Ladder L_4",       nx.ladder_graph(4),                   4),
 
        # ── Wheel graphs W_n = K_1 + C_{n-1} ────────────────────────────────
        ("Wheel W_5",        nx.wheel_graph(5),                    3),
        ("Wheel W_6",        nx.wheel_graph(6),                    4),
        ("Wheel W_7",        nx.wheel_graph(7),                    4),
        ("Wheel W_8",        nx.wheel_graph(8),                    4),
 
        # ── Small named graphs (brute-force verified) ─────────────────────────
        ("Paw (K_3+leaf)",
         nx.Graph([(0,1),(1,2),(2,0),(0,3)]),                      3),
        ("Diamond (K_4-e)",
         nx.Graph([(0,1),(1,2),(2,0),(0,3),(1,3)]),                3),
        ("Gem",              _gem(),                               4),
        ("Kite",             _kite(),                              4),
        ("House",            _house(),                             4),
        ("Bull",             nx.bull_graph(),                      3),
        ("Lollipop(3,2)",    nx.lollipop_graph(3, 2),             3),
        ("Friendship F_2",   _friendship(2),                       3),
        ("Friendship F_3",   _friendship(3),                       3),
 
        # ── Named benchmark graphs ────────────────────────────────────────────
        ("Petersen",         nx.petersen_graph(),                  4),
        ("Chvátal",          nx.chvatal_graph(),                   5),
        ("Mycielski χ=3",    nx.mycielski_graph(3),               3),
        ("Mycielski χ=4",    nx.mycielski_graph(4),               5),
 
    ]
 
    W           = 26
    model_names = ["repr1"]
    hdr = (f"{'Graph':<{W}} {'Γ':>4} "
           + " ".join(f"{m:>10}" for m in model_names)
           + f" {'Valid':>6} {'OK':>4}")
    print(hdr)
    print("─" * len(hdr))

    total, passed = 0, 0
    all_results   = []

    for name, G, chi_exp in tests:
        results, chis, valids = [], [], []
            
            
        r = repr_formulation2(
            G, 
            vertex_ordering=True, 
            phi_bound_constraint=True, 
            predecessor_bound_constraint=True,
            sum_phi_values_constraint=True,
            mtz_strengthening=True 
        )

        
        
        results.append(r)
        chis.append(r["gamma"])
        valids.append(r["valid"])

        agree     = all(c == chis[0] for c in chis)
        correct   = (chi_exp is None) or (chis[0] == chi_exp)
        valid_all = all(valids)
        ok        = agree and correct and valid_all
        total    += 1
        if ok:
            passed += 1

        exp_s   = str(chi_exp) if chi_exp else "?"
        chi_str = " ".join(f"{c:>10}" for c in chis)
        print(f"{name:<{W}} {exp_s:>4} {chi_str} "
              f"{'✓' if valid_all else '✗':>6} {'✓' if ok else '✗':>4}")
        all_results.append((name, G, chi_exp, results))

    print("─" * len(hdr))
    print(f"Result: {passed}/{total} tests passed\n")
    return all_results


def run_performance_tests() -> None:
    """Benchmark a list of solvers on larger instances and print a timing table.

    Runs each solver on a fixed set of graphs ranging from 10 to 30 vertices
    (Petersen, hypercube, Mycielski, and Erdős–Rényi at several densities)
    and prints wall-clock times in a formatted table.

    Parameters
    ----------
    solvers : list of (str, callable, callable, callable | None)
        Same format as :func:`run_correctness_tests`.

    Notes
    -----
    The Grundy number reported in each row is taken from the first solver in
    the list; all solvers are assumed to agree (verified by
    :func:`run_correctness_tests`).
    """


    perf = [
        
        ("ER n=10 p=0.1",  nx.erdos_renyi_graph(10, 0.1, seed=1)),
        ("ER n=10 p=0.2",  nx.erdos_renyi_graph(10, 0.2, seed=2)),
        ("ER n=10 p=0.3",  nx.erdos_renyi_graph(10, 0.3, seed=3)),
        ("ER n=10 p=0.4",  nx.erdos_renyi_graph(10, 0.4, seed=4)),
        ("ER n=10 p=0.5",  nx.erdos_renyi_graph(10, 0.5, seed=5)),
        ("ER n=10 p=0.6",  nx.erdos_renyi_graph(10, 0.6, seed=6)),
        ("ER n=10 p=0.7",  nx.erdos_renyi_graph(10, 0.7, seed=7)),
        ("ER n=10 p=0.8",  nx.erdos_renyi_graph(10, 0.8, seed=8)),
        ("ER n=10 p=0.9",  nx.erdos_renyi_graph(10, 0.9, seed=9)),
        

        ("ER n=15 p=0.1",  nx.erdos_renyi_graph(15, 0.1, seed=1)),
        ("ER n=15 p=0.2",  nx.erdos_renyi_graph(15, 0.2, seed=2)),
        ("ER n=15 p=0.3",  nx.erdos_renyi_graph(15, 0.3, seed=3)),
        ("ER n=15 p=0.4",  nx.erdos_renyi_graph(15, 0.4, seed=4)),
        ("ER n=15 p=0.5",  nx.erdos_renyi_graph(15, 0.5, seed=5)),
        ("ER n=15 p=0.6",  nx.erdos_renyi_graph(15, 0.6, seed=6)),
        ("ER n=15 p=0.7",  nx.erdos_renyi_graph(15, 0.7, seed=7)),
        ("ER n=15 p=0.8",  nx.erdos_renyi_graph(15, 0.8, seed=8)),
        ("ER n=15 p=0.9",  nx.erdos_renyi_graph(15, 0.9, seed=9)),
        
        ("ER n=20 p=0.1",  nx.erdos_renyi_graph(20, 0.1, seed=1)),
        ("ER n=20 p=0.2",  nx.erdos_renyi_graph(20, 0.2, seed=2)),
        ("ER n=20 p=0.3",  nx.erdos_renyi_graph(20, 0.3, seed=3)),
        ("ER n=20 p=0.4",  nx.erdos_renyi_graph(20, 0.4, seed=4)),
        ("ER n=20 p=0.5",  nx.erdos_renyi_graph(20, 0.5, seed=5)),
        ("ER n=20 p=0.6",  nx.erdos_renyi_graph(20, 0.6, seed=6)),
        ("ER n=20 p=0.7",  nx.erdos_renyi_graph(20, 0.7, seed=7)),
        ("ER n=20 p=0.8",  nx.erdos_renyi_graph(20, 0.8, seed=8)),
        ("ER n=20 p=0.9",  nx.erdos_renyi_graph(20, 0.9, seed=9)),
        
        ("ER n=25 p=0.1",  nx.erdos_renyi_graph(25, 0.1, seed=1)),
        ("ER n=25 p=0.2",  nx.erdos_renyi_graph(25, 0.2, seed=2)),
        ("ER n=25 p=0.3",  nx.erdos_renyi_graph(25, 0.3, seed=3)),
        ("ER n=25 p=0.4",  nx.erdos_renyi_graph(25, 0.4, seed=4)),
        ("ER n=25 p=0.5",  nx.erdos_renyi_graph(25, 0.5, seed=5)),
        ("ER n=25 p=0.6",  nx.erdos_renyi_graph(25, 0.6, seed=6)),
        ("ER n=25 p=0.7",  nx.erdos_renyi_graph(25, 0.7, seed=7)),
        ("ER n=25 p=0.8",  nx.erdos_renyi_graph(25, 0.8, seed=8)),
        ("ER n=25 p=0.9",  nx.erdos_renyi_graph(25, 0.9, seed=9)),
    ]    

    W           = 22
    model_names = ["rep" + str(i) for i in range(1<<3) ]
    hdr = (f"{'Graph':<{W}} {'Γ':>4} "
           + " ".join(f"{m:>12}" for m in model_names))
    print(hdr)
    print("─" * len(hdr))

    for name, G in perf:
        row_results = []
        for mask in range(1<<3):
            
            vertex_ordering = bool(mask & 1)
            phi_bound_constraint = bool(mask & 2)
            predecessor_bound_constraint = bool(mask & 4)
            
            #print("vertex_ordering ", vertex_ordering)
            #print("phi ", phi_bound_constraint)
            #print("predecessor  ", predecessor_bound_constraint)
            

            r = repr_formulation(
                G, 
                vertex_ordering, 
                phi_bound_constraint, 
                predecessor_bound_constraint, 
            )
    
            row_results.append(r)
        gamma = row_results[0]["gamma"]

        valid = all( r["gamma"] == row_results[0]["gamma"] for r in row_results)
        times = " ".join(f"{r['cpu_s']:>11.3f}s" for r in row_results)
        print(f"{name:<{W}} {gamma:>4} {times}" f"{'✓' if valid else '✗':>6}" )


if __name__ == "__main__":

    #run_correctness_tests()
    
    
    G = nx.erdos_renyi_graph(20, 0.25, 1)
    
    print("Modelo Base")
    #Modelo Base
    print( 
        repr_formulation2(
        G,
        vertex_ordering=False,
        phi_bound_constraint=False,
        predecessor_bound_constraint=False,
        sum_phi_values_constraint=False,
        mtz_strengthening=False,
        time_limit=600 # segundos
        )
    )
    print("Modelo Base + restrições")
    
    #Modelo Base com restrições novas
    print( 
        repr_formulation2(
        G,
        vertex_ordering=False,
        phi_bound_constraint=True,
        predecessor_bound_constraint=True,
        sum_phi_values_constraint=True,
        mtz_strengthening=True,
        time_limit=600 # segundos
        )
    )
    print("Modelo Base + ordenação SLO")
    
    # Modelo Base com a ordenação
    print( 
        repr_formulation2(
        G,
        vertex_ordering=True,
        phi_bound_constraint=False, # O(n)
        predecessor_bound_constraint=False, # O(n)
        sum_phi_values_constraint = False, # O(1)
        mtz_strengthening=False, #O(1)
        time_limit=600 # segundos
        )
    )
    print("Modelo Base + ordenação SLO + restrições")
    
    #Modelos Base com a ordenação + restrições
    print( 
        repr_formulation2(
        G,
        vertex_ordering=True,
        phi_bound_constraint=True, # O(n)
        predecessor_bound_constraint=True, # O(n)
        sum_phi_values_constraint = False, # O(1)
        mtz_strengthening=True, #O(1)
        time_limit=600 # segundos
        )
    )