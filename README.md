# Grundy Number — Branch-and-Bound Solvers and Coloring Enumerators

This repository provides exact solvers for the **Grundy (first-fit chromatic) number** of an undirected graph, together with an original enumerative algorithm that counts all distinct Grundy colorings without exhaustive permutation search.

---

## Table of Contents

1. [Background](#background)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
   - [Upper-bound functions](#upper-bound-functions)
   - [Revised stair factor (partial Grundy)](#revised-stair-factor-partial-grundy)
   - [Psi bound](#psi-bound)
   - [Vertex ordering strategies](#vertex-ordering-strategies)
   - [Lower-bound heuristics](#lower-bound-heuristics)
   - [Greedy coloring](#greedy-coloring)
   - [Greedy-coloring validator](#greedy-coloring-validator)
   - [Maximal clique enumeration](#maximal-clique-enumeration)
   - [Branch-and-bound solvers](#branch-and-bound-solvers)
   - [DSatur-based solvers](#dsatur-based-solvers)
   - [ILP formulations](#ilp-formulations)
   - [Enumerative algorithms](#enumerative-algorithms)
   - [Test harnesses](#test-harnesses)
6. [Running the Tests](#running-the-tests)
7. [References](#references)

---

## Background

Given an undirected graph $G = (V, E)$, a **greedy coloring** (also called a *first-fit coloring*) produced by a vertex ordering $\sigma = (v_1, v_2, \ldots, v_n)$ assigns to each vertex $v_i$ the smallest non-negative integer color not already used by any earlier neighbour $v_j$ ($j < i$, $v_j v_i \in E$).

The **Grundy number** $\Gamma(G)$ is the maximum number of colors used over all possible vertex orderings:

$$\Gamma(G) = \max_{\sigma \in S_n} \chi_{\text{greedy}}(G, \sigma)$$

It satisfies the classical chain of inequalities:

$$\chi(G) \leq \Gamma(G) \leq \Delta(G) + 1$$

where $\chi(G)$ is the chromatic number and $\Delta(G)$ is the maximum degree.

A coloring $C = (C_0, C_1, \ldots, C_{k-1})$ is a **Grundy coloring** (equivalently, it *can be produced by some greedy ordering*) if and only if every vertex $v \in C_i$ has at least one neighbour in each earlier class $C_j$ for $j < i$. This property is the foundation of all algorithms in this module.

---

## Repository Structure

```
grundy-number/
├── grundy/
│   ├── __init__.py          # Public API exports
│   └── bb_grundy.py         # All solvers and utilities
├── tests/
│   └── test_grundy.py       # pytest test suite
├── docs/
│   └── correctness.tex      # LaTeX proof of enumerate_orders correctness
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/grundy-number.git
cd grundy-number
pip install -e ".[dev]"
```

Dependencies: **Python ≥ 3.10**, **NetworkX ≥ 3.0**, **OR-Tools ≥ 9.0**.

---

## Quick Start

```python
import networkx as nx
from grundy import branch_and_bound3, dsatur_grundy2, counting_grundy_colorings
from grundy import strategy_smallest_last, fast_stair_factor

G = nx.petersen_graph()

# Exact Grundy number via branch-and-bound (recommended)
result = branch_and_bound3(G)
print(result["gamma"])    # 4
print(result["classes"])  # optimal Grundy coloring as list of lists

# DSatur-based solver with smallest-last ordering
order = strategy_smallest_last(G)
result2 = dsatur_grundy2(G, order=order, ub_func=fast_stair_factor)
print(result2["gamma"])   # 4

# Brute-force count of all distinct Grundy colorings (small graphs only)
G_small = nx.path_graph(6)
r = counting_grundy_colorings(G_small)
print(r["grundy_colorings"])  # total distinct Grundy colorings
```

---

## API Reference

### Upper-bound functions

| Function | Bound | Complexity | Notes |
|---|---|---|---|
| `delta_1(G)` | $\Delta(G) + 1$ | $O(n)$ | Trivial degree bound |
| `delta_2(G)` | $\max_{(u,v)\in E}\min(\deg u, \deg v)+1$ | $O(m)$ | Edge-based bound |
| `stair_factor(G)` | $\min_i (d_i + i)$ over residue sequence | $O(n^2)$ | Max-degree degeneracy ordering |
| `fast_stair_factor(G)` | same as above | $O(n + m)$ | Bucket-queue implementation |

The **stair factor** is computed from the *maximum-degree degeneracy ordering*: iteratively remove the vertex of highest residual degree, recording its degree as $d_i$ ($i = 1, 2, \ldots, n$, 1-indexed). The bound is $\min_i (d_i + i)$.

`delta_2` is a lightweight edge-based bound that can be tighter than `delta_1` on graphs with unbalanced degree sequences.

### Revised stair factor (partial Grundy)

The **revised stair factor** $\zeta'(G)$ extends the stair factor to upper-bound the *partial* Grundy number $\partial\Gamma(G)$, grouping maximum-degree vertices into equivalence classes by their open neighbourhood before each removal step.

| Function | Criterion | Complexity |
|---|---|---|
| `revised_stair_factor(G)` | min-node (reference) | $O(n(n+m))$ |
| `revised_stair_factor_fast(G)` | min-node (incremental) | $O(n+m)$ amortised |
| `revised_stair_factor2(G)` | max-size (reference) | $O(n(n+m))$ |
| `revised_stair_factor_fast2(G)` | max-size (incremental) | $O(n+m)$ amortised |

**Recommendation:** use `revised_stair_factor_fast2` in hot paths. The `max-size` criterion selects the largest neighbourhood class at each step, yielding bounds equal to or tighter than `min-node` in ~99.6% of tested graphs.

### Psi bound

| Function | Description | Complexity |
|---|---|---|
| `psi_bound(G)` | Scalar $\Psi(G) = \max_v \psi(v, \Delta+1)$ | $O(n \cdot \Delta^2)$ |
| `psi_bound2(G, ub)` | Full $\psi[v][k]$ table for all $v, k$ | $O(n \cdot \Delta^2)$ |

$\psi(v, k)$ is the largest $l$ such that there exist neighbours $u_1, \ldots, u_{l-1}$ of $v$ with $\psi(u_i, k) \geq i$. `psi_bound2` returns the complete table and is intended for use inside branch-and-bound nodes.

### Vertex ordering strategies

Vertex orderings control the search direction in DSatur-based solvers and the quality of greedy lower bounds.

| Function | Strategy | Complexity |
|---|---|---|
| `strategy_largest_first(G)` | Remove highest residual degree first | $O(n+m)$ |
| `strategy_smallest_last(G)` | Remove lowest residual degree last (degeneracy) | $O(n+m)$ |
| `reverse_smallest_last(G)` | Reverse of the above | $O(n+m)$ |
| `strategy_smallest_dsatur(G)` | Minimum-saturation first, degree tie-break | $O(n^2+m)$ |
| `strategy_smallest_dsatur_bucket(G)` | Same, bucket-queue | $O(n\log n+m)$ |
| `coloring_order(G)` | Order induced by the best greedy lower-bound coloring | $O(n+m)$ |

### Lower-bound heuristics

| Function | Strategy |
|---|---|
| `lower_bound(G)` | Reverse smallest-last ordering → greedy coloring |
| `lower_bound2(G)` | Reverse largest-first ordering → greedy coloring |

Both return `{"coloring": list[list], "lower_bound": int}`. Any greedy coloring is a valid Grundy coloring, so both provide valid lower bounds.

### Greedy coloring

```python
colors: dict = greedy_coloring(G, nodes)
```

Assigns to each vertex the minimum-excludant (mex) color given the already-colored neighbours. Runs in $O(n + m)$.

### Greedy-coloring validator

```python
ok: bool = is_greedy_coloring(G, C)
```

Returns `True` if and only if every vertex $v \in C_i$ has at least one neighbour in $C_j$ for every $j < i$. This is the defining property of a Grundy coloring. Also checks that each class is an independent set.

### Maximal clique enumeration

```python
cliques: generator = find_cliques(G, nodes=None)
```

Yields all maximal cliques of `G` using the Bron–Kerbosch algorithm with Tomita–Tanaka–Takahashi pivot selection, implemented iteratively to avoid Python recursion limits. When called on the complement graph $\bar{G}$, each clique corresponds to a maximal independent set of $G$, which is how the branch-and-bound solvers use it.

Worst-case complexity: $O(3^{n/3})$ (Moon–Moser bound).

### Branch-and-bound solvers

All three solvers implement the recurrence:

$$\Gamma(S) = \max \{ \Gamma(S \setminus X) + 1 \mid X \subseteq S \text{ is a maximal independent set of } G[S] \}$$

They differ only in how the subgraph is maintained across recursive calls.

| Function | Subgraph maintenance | Upper bound |
|---|---|---|
| `branch_and_bound(G, ub_func)` | Full graph copy | pluggable |
| `branch_and_bound2(G, ub_func)` | Frozenset views | pluggable |
| `branch_and_bound3(G, ub_func)` | In-place mutation + backtrack | pluggable |

All three return:

```python
{
    "model":    str,         # solver identifier
    "gamma":    int,         # Grundy number Γ(G)
    "classes":  list[list],  # optimal coloring
    "valid":    bool,        # is_greedy_coloring(G, classes)
    "cpu_s":    float,       # wall-clock seconds
    "bb_nodes": int,         # branch-and-bound nodes expanded
}
```

**Recommendation:** use `branch_and_bound3` for general use; it avoids allocation overhead at each node via in-place mutation with backtracking. Note that it is **not thread-safe** as it mutates the graph in place.

Convenience wrappers with fixed bounds are available as `branch(G)`, `branch2(G)`, and `branch3(G)`.

### DSatur-based solvers

These solvers build Grundy colorings incrementally, assigning each vertex in a fixed *order* to a colour class, with backtracking. They differ in the pruning mechanisms applied.

| Function | Pruning | Complexity |
|---|---|---|
| `dsatur_grundy(G, ub_func)` | Independence constraint only | Exponential |
| `dsatur_grundy2(G, order, ub_func)` | + Symmetry breaking + local $\delta_2$ bounds | Exponential, faster |
| `dsatur_grundy3(G, order, ub_func)` | + Incremental MEX + availability cuts | Exponential, fastest |

`dsatur_grundy2` and `dsatur_grundy3` require an explicit vertex ordering. Typical choices are `strategy_smallest_last(G)` and `strategy_largest_first(G)`.

All three return:

```python
{
    "model":             str,
    "gamma":             int,
    "classes":           list[list],
    "valid":             bool,
    "grundy_colorings":  int,   # distinct valid Grundy colorings found
    "total":             int,   # leaf nodes explored
    "bb_nodes":          int,
    "cpu_s":             float,
    # dsatur_grundy3 also includes:
    "cuts":              int,   # subtrees pruned by availability cut
}
```

**Local helper:** `delta2(G, u)` computes $\delta_2(u) = \max\{\deg(w) \mid w \in N(u),\, \deg(w) < \deg(u)\} + 1$, used as a per-vertex upper bound inside the DSatur solvers.

### ILP formulations

All ILP solvers use **OR-Tools / SCIP** and return:

```python
{
    "model":              str,
    "gamma":              int | None,
    "optimal":            bool,
    "cpu_s":              float,
    "classes":            list[list],
    "valid":              bool,
    "linear_relaxation":  float,
}
```

| Function | Formulation | Grundy constraints | Variables |
|---|---|---|---|
| `solver_rodrigues(G, ub)` | Partition [Rod20] | Linear $O(\|V\|\cdot\|C\|)$ | $x[v,c]$, $z[c]$ |
| `solver_carvalho(G, ub)` | Partition [Car23] F2 | Quadratic $O(\|V\|\cdot\|C\|^2)$ | $x[v,c]$, $z[c]$ |
| `solver_carvalho_modificado(G, ub)` | Partition + aggregated cut | $O(\|V\|\cdot\|C\|^2)$ | $x[v,c]$, $z[c]$ |
| `solver_carvalho_representante(G)` | Representative [Car23] F3 | $O(\|V\|^2\cdot\|C\|)$ | $x[u,v]$, $y[u,v]$, pot |
| `solver_carvalho_representante2(G, ub)` | Representative, explicit UB | same | same |
| `solver_carvalho_representante3(G, order, ub)` | Representative, custom order | same | same |

**`get_linear_relaxation(solver)`** computes the LP relaxation value of any built OR-Tools model by temporarily relaxing all integer variables, solving, and restoring integrality.

### Enumerative algorithms

```python
result: dict = counting_grundy_colorings(G)
```

Iterates over all $n!$ permutations, applies `greedy_coloring`, validates with `is_greedy_coloring`, and collects distinct colorings. Returns the same keys as the B&B solvers plus `"grundy_colorings"` (total distinct Grundy colorings). Feasible only for $n \leq 10$–12. Intended as a correctness reference.

### Test harnesses

```python
results = run_correctness_tests(solvers)
run_performance_tests(solvers)
```

`run_correctness_tests` runs a battery of ~50 graphs with known Grundy numbers (paths, cycles, complete graphs, bipartite graphs, crowns, hypercubes, grids, wheels, Petersen, and Erdős–Rényi random graphs) and prints a table comparing all solvers. A test is marked ✓ if all solvers agree, the result matches the expected value, and all colorings pass `is_greedy_coloring`.

`run_performance_tests` benchmarks solvers on larger Erdős–Rényi instances ($n \in \{10, 15, 20, 25\}$, $p \in \{0.1, \ldots, 0.9\}$) and prints wall-clock times.

Each solver entry is a 4-tuple `(name, solver_fn, ub_fn, order_fn)`.

---

## Running the Tests

```bash
pytest tests/ -v
```

The test suite verifies:
- Correctness of `greedy_coloring` and `is_greedy_coloring` on path, cycle, complete, and random graphs.
- All branch-and-bound solvers agree with known Grundy numbers.
- DSatur solvers (`dsatur_grundy2`, `dsatur_grundy3`) agree with B&B on correctness instances.
- `counting_grundy_colorings` matches brute-force permutation count on small random graphs (seeds 1–8).
- Every partition returned by any solver passes `is_greedy_coloring`.

---

## References

- **[BGK05]** Shi, Z., Goddard, W., Hedetniemi, S. T., Kennedy, K., Laskar, R., & McRae, A. (2005). An algorithm for partial Grundy number on trees. *Discrete Mathematics*, 304(1–3), 108–116.
- **[BFK18]** Bonnet, É., Foucaud, F., Kim, E. J., & Sikora, F. (2018). Complexity of Grundy coloring and its variants. *Discrete Applied Mathematics*, 243, 99–114.
- **[BK73]** Bron, C. and Kerbosch, J. (1973). Algorithm 457: finding all cliques of an undirected graph. *Communications of the ACM*, 16(9), 575–577.
- **[TTT06]** Tomita, E., Tanaka, A., & Takahashi, H. (2006). The worst-case time complexity for generating all maximal cliques and computational experiments. *Theoretical Computer Science*, 363(1), 28–42.
- **[CK08]** Cazals, F., & Karande, C. (2008). A note on the problem of reporting maximal cliques. *Theoretical Computer Science*, 407(1–3), 564–568.
- **[MM65]** Moon, J. and Moser, L. (1965). On cliques in graphs. *Israel Journal of Mathematics*, 3(1), 23–28.
- **[Rod20]** Rodrigues, E. N. H. D. (2020). Coloração k-imprópria gulosa. Repositório UFC.
- **[Car23]** Carvalho, M., Melo, R., Santos, M. C., Toso, R. F., and Resende, M. G. C. (2023). Formulações de programação inteira para o problema da coloração de Grundy. Anais SBPO 2024.
- **[PV19]** Panda, B. S. and Verma, S. (2019). On partial Grundy coloring of bipartite graphs and chordal graphs. *Discrete Applied Mathematics*, 271, 171–183.
