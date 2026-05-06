"""
slo_vs_random_experiment.py
===========================
Para cada configuração de grafo (n, p), gera num_graphs grafos aleatórios
e compara a SLO contra N_RAND=500 ordens aleatórias nas métricas:

  - cpu_s              (tempo de resolução — menor é melhor)
  - linear_relaxation  (valor da relaxação LP — maior é melhor)
  - n_constraints      (número de restrições — menor é melhor)
  - nodes_explored     (nós explorados no B&B — menor é melhor)

Para cada métrica e cada grafo calcula:
  - valor da SLO
  - média, desvio padrão, mín e máx da distribuição aleatória
  - p-valor empírico unilateral
  - z-score
  - Δ = μ_rnd - valor_SLO  (positivo = SLO melhor)

Ao final aplica o Método de Fisher para combinar os p-valores por
configuração e por métrica.

Dependências: networkx, scipy, ortools, numpy
(todas as funções de formulação são importadas do arquivo original)
"""

import json
import math
import time
import random
import pathlib
import statistics
from datetime import datetime
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.stats import norm as sp_norm
from scipy.stats import chi2 as sp_chi2

from collections import defaultdict as _dd

def slo_fn(G: nx.Graph) -> list:
    """Smallest-Last Ordering para networkx.Graph."""
    nodes  = list(G.nodes())
    n      = len(nodes)
    degree = {v: G.degree(v) for v in nodes}
    removed = set()
    removal_order = []
    buckets = _dd(set)
    for v, d in degree.items():
        buckets[d].add(v)
    for _ in range(n):
        for d in range(n):
            if buckets[d] - removed:
                v = next(iter(buckets[d] - removed))
                buckets[d].discard(v)
                break
        removed.add(v)
        removal_order.append(v)
        for u in G.neighbors(v):
            if u not in removed:
                old_d = degree[u]
                buckets[old_d].discard(u)
                degree[u] -= 1
                buckets[degree[u]].add(u)
    return list(reversed(removal_order))   # menor índice = maior grau residual


# ── constantes ────────────────────────────────────────────────────────────
N_RAND      = 500          # permutações aleatórias por grafo
NUM_GRAPHS  = 10           # grafos por configuração
SAVE_DIR    = "results_slo_experiment"
TIME_LIMIT  = 600000      # ms (2 min por solve)

# Configurações (n, p)
CONFIGS = [
    (15, 0.1),
    (15, 0.25),
    (15, 0.5),
    (15, 0.75),
    (15, 0.9),
]

# Métricas e se "menor é melhor"
METRICS = {
    "cpu_s":             True,   # menor é melhor
    "linear_relaxation": True,  # maior é melhor (relaxação mais apertada)
    "n_constraints":     True,   # menor é melhor
    "nodes_explored":    True,   # menor é melhor
}

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

from greedy_coloring import is_greedy_coloring
import networkx as nx
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from upper_bound import upper_bound1


def repr_formulation(
    grafo: nx.Graph,
    order: list[int],
    time_limit: int = math.inf,
) -> dict:
    nodes  = list(grafo.nodes)
    n      = len(nodes)
    pos    = {v: i for i, v in enumerate(order)}
    adj    = {u: set(grafo.adj[u]) for u in nodes}

    upperbound = upper_bound1(grafo)

    # antiG[u]    : non-neighbours of u (candidates to share u's class)
    # antiGcol[u] : antiG[u] ∪ {u}  (u can also represent itself)
    # anti_set[u] : set version of antiGcol[u] for O(1) membership tests
    antiG    = {u: [v for v in nodes if v != u and v not in adj[u]] for u in nodes}
    antiGcol = {u: antiG[u] + [u] for u in nodes}
    anti_set = {u: set(antiGcol[u]) for u in nodes}

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
                solver.Add(phi[u] - phi[v] + 1 <= upperbound * (1 - y[u, v]))
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
        "model":             "rep with order",
        "gamma":             gamma,
        "optimal":           status == pywraplp.Solver.OPTIMAL,
        "cpu_s":             cpu,
        "classes":           classes,
        "valid":             valid,
        "linear_relaxation": lp_val,
        "nodes_explored":    solver.nodes(),
        "n_constraints":     solver.NumConstraints(),
        "lp_gap":            lp_gap,
    }



# ── helpers ───────────────────────────────────────────────────────────────

def random_order(G: nx.Graph, seed: int) -> list:
    nodes = list(G.nodes())
    rng   = random.Random(seed)
    rng.shuffle(nodes)
    return nodes


def p_value(r_slo, r_rnds, lower_is_better: bool) -> float:
    """Fração de ordens aleatórias tão boas ou melhores que a SLO."""
    if lower_is_better:
        return sum(1 for r in r_rnds if r <= r_slo) / len(r_rnds)
    else:
        return sum(1 for r in r_rnds if r >= r_slo) / len(r_rnds)


def z_score(r_slo, mu, sd, lower_is_better: bool) -> float:
    if sd == 0:
        return float("inf") if r_slo != mu else 0.0
    if lower_is_better:
        return (mu - r_slo) / sd    # positivo = SLO melhor
    else:
        return (r_slo - mu) / sd    # positivo = SLO melhor


def fisher_combined_p(p_values: list) -> float:
    """Método de Fisher para combinar p-valores independentes."""
    p_vals = [max(p, 1e-10) for p in p_values]   # evita log(0)
    chi2   = -2 * sum(math.log(p) for p in p_vals)
    df     = 2 * len(p_vals)
    return float(sp_chi2.sf(chi2, df))


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "ns "


def safe_get(d: dict, key: str):
    v = d.get(key)
    return v if v is not None else float("nan")


# ── experimento principal ─────────────────────────────────────────────────

def run_config(n: int, p: float) -> list:
    """Roda NUM_GRAPHS grafos para a config (n, p). Retorna lista de records."""
    print(f"\n{'='*70}")
    print(f"  CONFIG: n={n}, p={p}  |  {NUM_GRAPHS} grafos × {N_RAND} aleatórias")
    print(f"{'='*70}")

    header = (
        f"{'g':>3} {'metric':<22} {'SLO':>9} {'μ_rnd':>9} "
        f"{'σ_rnd':>8} {'z':>7} {'p-val':>7} {'sig':>4} {'Δ':>9}"
    )
    print(header)
    print("-" * 85)

    records = []

    for gi in range(NUM_GRAPHS):
        seed = 42 + gi * 97 + n * 1000 + int(p * 100) * 10000
        G    = nx.erdos_renyi_graph(n, p, seed=seed)

        # ── SLO ──────────────────────────────────────────────────────────
        order_slo = slo_fn(G)
        res_slo   = repr_formulation(G, order_slo, time_limit=TIME_LIMIT)

        slo_vals = {
            "cpu_s":             safe_get(res_slo, "cpu_s"),
            "linear_relaxation": safe_get(res_slo, "linear_relaxation"),
            "n_constraints":     safe_get(res_slo, "n_constraints"),
            "nodes_explored":    safe_get(res_slo, "nodes_explored"),
        }

        # ── 500 ordens aleatórias ────────────────────────────────────────
        rand_vals = defaultdict(list)
        for k in range(N_RAND):
            order_rnd = random_order(G, seed=seed + k * 13 + 7777)
            res_rnd   = repr_formulation(G, order_rnd, time_limit=TIME_LIMIT)
            for m in METRICS:
                rand_vals[m].append(safe_get(res_rnd, m))

        # ── estatísticas por métrica ─────────────────────────────────────
        metric_stats = {}
        for m, lower_better in METRICS.items():
            v_slo  = slo_vals[m]
            rnds   = [r for r in rand_vals[m] if not math.isnan(r)]
            if not rnds or math.isnan(v_slo):
                metric_stats[m] = None
                continue

            mu  = statistics.mean(rnds)
            sd  = statistics.stdev(rnds) if len(rnds) > 1 else 0.0
            mn  = min(rnds)
            mx  = max(rnds)
            pv  = p_value(v_slo, rnds, lower_better)
            z   = z_score(v_slo, mu, sd, lower_better)
            # Δ sempre positivo = SLO melhor
            delta = (mu - v_slo) if lower_better else (v_slo - mu)

            metric_stats[m] = {
                "slo":    v_slo,
                "mu":     mu,
                "sd":     sd,
                "min":    mn,
                "max":    mx,
                "p_val":  pv,
                "z":      z,
                "delta":  delta,
                "rand_vals": rnds,
            }

            print(
                f"{gi:>3} {m:<22} {v_slo:>9.3f} {mu:>9.3f} "
                f"{sd:>8.3f} {z:>7.3f} {pv:>7.4f} {sig_stars(pv):>4} "
                f"{delta:>+9.3f}"
            )

        records.append({
            "graph_index": gi,
            "n": n, "p": p,
            "n_edges": G.number_of_edges(),
            "gamma": res_slo.get("gamma"),
            "slo_vals": slo_vals,
            "rand_vals": {m: list(v) for m, v in rand_vals.items()},
            "metric_stats": metric_stats,
        })
        print()

    return records


def summarize_config(records: list, n: int, p: float) -> dict:
    """Agrega estatísticas e aplica Fisher por métrica."""
    print(f"\n{'─'*70}")
    print(f"  RESUMO CONFIG n={n}, p={p}  ({len(records)} grafos)")
    print(f"{'─'*70}")
    print(f"{'metric':<22} {'Δ médio':>10} {'z médio':>8} "
          f"{'%SLO>':>8} {'Fisher p':>10} {'sig':>4}")
    print("-" * 65)

    summary_metrics = {}
    for m, lower_better in METRICS.items():
        stats_list = [r["metric_stats"][m] for r in records
                      if r["metric_stats"].get(m) is not None]
        if not stats_list:
            continue

        deltas = [s["delta"] for s in stats_list]
        zs     = [s["z"]     for s in stats_list]
        pvals  = [s["p_val"] for s in stats_list]
        frac_better = sum(1 for s in stats_list if s["delta"] > 0) / len(stats_list)

        fisher_p = fisher_combined_p(pvals)
        mean_delta = statistics.mean(deltas)
        mean_z     = statistics.mean(zs)

        print(
            f"{m:<22} {mean_delta:>+10.3f} {mean_z:>8.3f} "
            f"{100*frac_better:>7.1f}% {fisher_p:>10.4f} {sig_stars(fisher_p):>4}"
        )

        summary_metrics[m] = {
            "mean_delta":   mean_delta,
            "mean_z":       mean_z,
            "frac_better":  frac_better,
            "fisher_p":     fisher_p,
            "individual_pvals": pvals,
        }

    return summary_metrics


# ── ponto de entrada ──────────────────────────────────────────────────────

if __name__ == "__main__":
    pathlib.Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for n, p in CONFIGS:
        records = run_config(n, p)
        summary = summarize_config(records, n, p)
        all_results[f"n{n}_p{int(p*100)}"] = {
            "n": n, "p": p,
            "records": records,
            "summary": summary,
        }

    # ── tabela final cruzada ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  TABELA FINAL — Fisher p-valor por (n, p) e métrica")
    print(f"{'='*80}")

    metrics_list = list(METRICS.keys())
    header = f"{'(n,p)':<12}" + "".join(f"{m[:14]:>16}" for m in metrics_list)
    print(header)
    print("-" * 80)

    for key, data in all_results.items():
        row = f"({data['n']},{data['p']:<4})"
        for m in metrics_list:
            s = data["summary"].get(m)
            if s:
                cell = f"{s['fisher_p']:.4f}{sig_stars(s['fisher_p'])}"
            else:
                cell = "  N/A  "
            row += f"{cell:>16}"
        print(row)

    print()
    print("Legenda: *** p<0.001  ** p<0.01  * p<0.05  ns=não significativo")
    print("Δ>0 em todas as métricas significa SLO melhor que a média aleatória.")

    # ── salvar JSON ───────────────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = pathlib.Path(SAVE_DIR) / f"slo_experiment_{ts}.json"

    # serializar removendo rand_vals detalhadas para não inflar o JSON
    for key in all_results:
        for rec in all_results[key]["records"]:
            rec.pop("rand_vals", None)   # remove listas de 500 itens

    with open(fname, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResultados salvos em: {fname}")
