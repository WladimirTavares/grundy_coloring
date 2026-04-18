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
   - [Lower-bound heuristics](#lower-bound-heuristics)
   - [Greedy coloring](#greedy-coloring)
   - [Greedy-coloring validator](#greedy-coloring-validator)
   - [Branch-and-bound solvers](#branch-and-bound-solvers)
   - [Enumerative baseline](#enumerative-baseline)
   - [enumerate_orders](#enumerate_orders)
6. [The enumerate_orders Algorithm](#the-enumerate_orders-algorithm)
   - [Motivation](#motivation)
   - [The four moves](#the-four-moves)
   - [Pending evictions and min_class](#pending-evictions-and-min_class)
   - [Correctness argument](#correctness-argument)
   - [Complexity](#complexity)
7. [Running the Tests](#running-the-tests)
8. [References](#references)

---

## Background

Given an undirected graph $G = (V, E)$, a **greedy coloring** (also called a *first-fit coloring*) produced by a vertex ordering $\sigma = (v_1, v_2, \ldots, v_n)$ assigns to each vertex $v_i$ the smallest non-negative integer color not already used by any earlier neighbour $v_j$ ($j < i$, $v_j v_i \in E$).

The **Grundy number** $\Gamma(G)$ is the maximum number of colors used over all possible vertex orderings:

$$\Gamma(G) = \max_{\sigma \in S_n} \chi_{\text{greedy}}(G, \sigma)$$

It satisfies the classical chain of inequalities:

$$\chi(G) \leq \Gamma(G) \leq \Delta(G) + 1$$

where $\chi(G)$ is the chromatic number and $\Delta(G)$ is the maximum degree.

A coloring $C = (C_0, C_1, \ldots, C_{k-1})$ is a **Grundy coloring** (equivalently, it *can be produced by some greedy ordering*) if and only if every vertex $v \in C_i$ has at least one neighbour in each earlier class $C_j$ for $j < i$. This property is the foundation of both the branch-and-bound recurrence and the `enumerate_orders` algorithm.

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

Dependencies: **Python ≥ 3.10**, **NetworkX ≥ 3.0**.

---

## Quick Start

```python
import networkx as nx
from grundy import branch3, enumerate_orders, counting_grundy_colorings

G = nx.petersen_graph()

# Exact Grundy number via branch-and-bound
result = branch3(G)
print(result["gamma"])    # e.g. 4
print(result["classes"])  # optimal Grundy coloring as list of lists

# Count all distinct Grundy colorings (small graphs only)
G_small = nx.path_graph(6)
n_bf = counting_grundy_colorings(G_small)  # brute-force baseline
n_eo = len(enumerate_orders(G_small))      # enumerative algorithm
assert n_bf == n_eo
```

---

## API Reference

### Upper-bound functions

| Function | Bound | Complexity |
|---|---|---|
| `upper_bound(G)` | $\Delta(G) + 1$ | $O(n)$ |
| `stair_factor(G)` | $\min_i (d_i + i)$ over residue sequence | $O(n^2)$ |
| `fast_stair_factor(G)` | same as above, bucket-queue | $O(n + m)$ |

The **stair factor** is computed from the *maximum-degree degeneracy ordering*: iteratively remove the vertex of highest residual degree, recording its degree as $d_i$ ($i = 1, 2, \ldots, n$, 1-indexed). The bound is $\min_i (d_i + i)$, which is always at most $\Delta(G) + 1$ and often significantly tighter.

### Lower-bound heuristics

| Function | Strategy |
|---|---|
| `lower_bound(G)` | Reverse smallest-last ordering → greedy coloring |
| `lower_bound2(G)` | Reverse largest-first ordering → greedy coloring |

Both return a valid lower bound because any greedy coloring is a valid Grundy coloring.

### Greedy coloring

```python
colors: dict = greedy_coloring(G, nodes)
```

Assigns to each vertex the minimum-excludant (mex) color given the already-colored neighbours. Runs in $O(n + m)$.

### Greedy-coloring validator

```python
ok: bool = is_greedy_coloring(G, C)
```

Returns `True` if and only if every vertex $v \in C_i$ has at least one neighbour in $C_j$ for every $j < i$. This is the defining property of a Grundy coloring.

### Branch-and-bound solvers

All three solvers implement the recurrence:

$$\Gamma(S) = \max\bigl\{\, \Gamma(S \setminus X) + 1 \;\big|\; X \subseteq S \text{ is a maximal independent set of } G[S] \bigr\}$$

They differ only in the upper-bound function used for pruning.

| Function | Upper bound | Node cost |
|---|---|---|
| `branch(G)` | $\Delta(G[S]) + 1$ | $O(n)$ |
| `branch2(G)` | stair factor | $O(n^2)$ |
| `branch3(G)` | fast stair factor | $O(n + m)$ |

All three return a dictionary:

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

**Recommendation:** use `branch3` for general use. Use `branch` only when profiling overhead of the bound computation matters.

### Enumerative baseline

```python
result: dict = enumerate_grundy_colorings(G)   # finds the maximum only
count:  int  = counting_grundy_colorings(G)    # counts all distinct Grundy colorings
```

Both iterate over all $n!$ permutations and are intended only for small graphs ($n \leq 12$). They serve as a correctness reference for `enumerate_orders`.

### enumerate_orders

```python
colorings: set = enumerate_orders(G)
```

Returns the set of all distinct Grundy colorings of $G$ as a set of tuples of sorted tuples. See the next section for a full description of the algorithm.

---

## The enumerate_orders Algorithm

### Motivation

The brute-force approach (`counting_grundy_colorings`) iterates over all $n!$ vertex orderings, which is infeasible for $n \gtrsim 14$. Many orderings produce the same color-class partition, and many partitions are clearly not Grundy colorings.

`enumerate_orders` inverts the perspective: instead of iterating over orderings and deriving partitions, it directly constructs color-class partitions incrementally via backtracking, verifying the Grundy condition only at the leaves. This avoids redundant permutations and admits structural pruning.

### The four moves

The algorithm places vertices one by one. The *current vertex* $v$ is either the first element of the `pending` list (an evicted vertex, described below) or `remaining[idx]` (the next unplaced vertex from the fixed input order). For each existing color class $C[i]$ (iterated from `min_class` upward), the algorithm tries four moves:

**Move 1 — No conflict.**
If $v$ has no neighbour in $C[i]$, append $v$ to $C[i]$. This corresponds to a greedy ordering where $v$ is colored *after* all current members of $C[i]$ and picks color $i$ because no conflict exists.

**Move 2 — New class immediately after $i$.**
Insert the singleton $\{v\}$ as a new class at position $i+1$. This corresponds to a greedy ordering where $v$ is colored after the members of $C[i]$ but gets a higher color because of a conflict.

**Move 3 — Eviction.**
$v$ enters $C[i]$, and its neighbours in $C[i]$ are *evicted* into the `pending` list with `min_class = i + 1`. This simulates a greedy ordering where $v$ is colored *before* those neighbours: when $v$ is processed, it picks color $i$ freely, and when the evicted neighbours are later processed, they find $v$ in $C[i]$ and are forced to a higher color. The evicted vertices can validly form a class at any position $j > i$ because they are a subset of the independent set $C[i]$.

**Move 4 — New class at the end.**
Open a fresh singleton $\{v\}$ appended at the end of the current partition. This is always a valid option regardless of conflicts.

After all four moves are tried for all existing classes, any completed partition (no vertices unplaced, no pending evictions) is verified by `is_greedy_coloring` and added to the result set if valid.

### Pending evictions and min_class

The `pending` list carries pairs `(vertex, min_class)`. The `min_class` constraint ensures that an evicted vertex is never re-inserted into the class it was removed from, nor into any earlier class. The loop over existing classes begins at `min_class`:

```python
for i in range(min_class, len(C)):
    ...
```

When an evicted vertex itself triggers another eviction (Move 3 applied to a pending vertex), the new evicted vertices receive `min_class = i + 1` where `i` is the *current* class position, not inherited from the parent eviction. This is correct: what matters is only the position of the class from which they are removed.

### Correctness argument

**Claim:** `enumerate_orders(G)` returns exactly the set of all Grundy colorings of $G$.

**Soundness** (no false positives): every partition added to the result set passes `is_greedy_coloring`, which checks the Grundy condition exactly. Therefore every returned partition is indeed a Grundy coloring.

**Completeness** (no false negatives): let $\mathcal{C} = (C_0, C_1, \ldots, C_{k-1})$ be any Grundy coloring of $G$, produced by some ordering $\sigma$. We show that `enumerate_orders` generates $\mathcal{C}$ (or a partition equal to it as a set-partition).

Process the vertices of $\sigma$ in order. At the moment vertex $v$ is processed by the greedy rule under $\sigma$, it receives color $i$ — meaning $v$ has at least one neighbour in each of $C_0, \ldots, C_{i-1}$ and no neighbour in $C_i$ (among the vertices already colored). In the backtrack, this situation is captured by one of the four moves:

- If $v$ has no neighbour in $C[i]$ at the current state of the backtrack → **Move 1**.
- If $v$ is forced to a new class due to conflicts with $C[i]$ → **Move 2** or **Move 4**.
- If some members of $C[i]$ were colored *after* $v$ in $\sigma$ and therefore $v$ should enter $C[i]$ displacing them → **Move 3** (eviction).

Because the backtrack explores all combinations of these moves across all classes and all vertices, it will reach a state where $C = \mathcal{C}$. The pruning via `stair_factor` only discards partial states where the number of open classes already exceeds the stair-factor upper bound on $\Gamma(G)$; since $|\mathcal{C}| = k \leq \Gamma(G) \leq \text{stair\_factor}(G)$, no valid Grundy coloring is pruned.

**Formal proof:** See `docs/correctness.tex` for a full inductive proof.

### Complexity

In the worst case the number of candidate partitions explored is super-exponential (the Bell number $B_n$ bounds the number of set-partitions). In practice the algorithm is far faster than the $n!$ brute-force baseline because:

1. Many structurally equivalent orderings are merged into a single partition (de-duplication via the set).
2. The `stair_factor` pruning eliminates subtrees that cannot produce colorings with more classes than the current upper bound.
3. The `min_class` constraint on pending vertices reduces branching for evicted vertices.

---

## Running the Tests

```bash
pytest tests/ -v
```

The test suite verifies:
- Correctness of `greedy_coloring` and `is_greedy_coloring` on path, cycle, complete, and random graphs.
- All three branch-and-bound solvers agree with known Grundy numbers.
- `enumerate_orders` matches `counting_grundy_colorings` on small random graphs (seeds 1–8).
- Every partition returned by `enumerate_orders` passes `is_greedy_coloring`.

---

## References

- **[BGK05]** Shi, Z., Goddard, W., Hedetniemi, S. T., Kennedy, K., Laskar, R., & McRae, A. (2005). An algorithm for partial Grundy number on trees. *Discrete Mathematics*, 304(1–3), 108–116.
- **[BFK18]** Bonnet, É., Foucaud, F., Kim, E. J., & Sikora, F. (2018). Complexity of Grundy coloring and its variants. *Discrete Applied Mathematics*, 243, 99–114.
- **[BK73]** Bron, C. and Kerbosch, J. (1973). Algorithm 457: finding all cliques of an undirected graph. *Communications of the ACM*, 16(9), 575–577.
- **[TTT06]** Tomita, E., Tanaka, A., & Takahashi, H. (2006). The worst-case time complexity for generating all maximal cliques and computational experiments. *Theoretical Computer Science*, 363(1), 28–42.
- **[CK08]** Cazals, F., & Karande, C. (2008). A note on the problem of reporting maximal cliques. *Theoretical Computer Science*, 407(1–3), 564–568.
- **[MM65]** Moon, J. and Moser, L. (1965). On cliques in graphs. *Israel Journal of Mathematics*, 3(1), 23–28.
