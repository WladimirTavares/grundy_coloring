"""
experiment_analysis.py
======================
Módulo com:
  - run_experiment_v2   : versão estendida que persiste todos os dados em JSON
  - plot_all_visualizations : gera as 5 visualizações discutidas
  - load_results        : carrega resultados salvos para análises futuras

Dependências: networkx, numpy, scipy, matplotlib, seaborn, json, pathlib
"""

import json
import math
import time
import random
import pathlib
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Paleta e estilo globais
# ---------------------------------------------------------------------------
PALETTE = {
    "heuristic":  "#E84855",   # vermelho
    "random":     "#3A86FF",   # azul
    "accent":     "#FFBE0B",   # amarelo
    "bg":         "#0F0F14",
    "panel":      "#1A1A24",
    "grid":       "#2A2A38",
    "text":       "#E8E8F0",
    "text_muted": "#7070A0",
}

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text_muted"],
        "ytick.color":       PALETTE["text_muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["grid"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.facecolor":  PALETTE["panel"],
        "legend.edgecolor":  PALETTE["grid"],
        "font.family":       "monospace",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })


# ---------------------------------------------------------------------------
# Helpers estatísticos
# ---------------------------------------------------------------------------

def percentile_lower_is_better(values, x):
    return sum(v >= x for v in values) / len(values) * 100

def percentile_higher_is_better(values, x):
    return sum(v <= x for v in values) / len(values) * 100

def _rolling_corr(x_series, y_series, window=10):
    corrs = [None] * (window - 1)
    for i in range(window - 1, len(x_series)):
        xw = x_series[i - window + 1: i + 1]
        yw = y_series[i - window + 1: i + 1]
        if np.std(xw) == 0 or np.std(yw) == 0:
            corrs.append(0.0)
        else:
            corrs.append(pearsonr(xw, yw)[0])
    return corrs


def solver_carvalho_representante2(
    grafo: nx.Graph,
    order: list[int],
    time_limit: int = math.inf,
) -> dict:
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


def run_experiment_v3(
    heuristic_function,
    num_graphs: int = 50,
    n: int = 50,
    p: float = 0.5,
    k_perms: int = 200,
    save_dir: str = "results",
    experiment_tag: str = "",
    solver_fn=None,
    stair_factor_fn=None,
    sample_permutations_fn=None,
):
    import networkx as nx

    assert solver_fn is not None
    assert stair_factor_fn is not None
    assert sample_permutations_fn is not None

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    records = []
    percentis_time        = []
    percentis_relaxation  = []
    percentis_nodes       = []
    percentis_nvars       = []
    percentis_lp_gap      = []

    for i in range(num_graphs):
        G     = nx.erdos_renyi_graph(n, p)
        nodes = list(G.nodes())
        sf    = stair_factor_fn(G)

        # --- heurística ---
        r_h   = solver_fn(G, heuristic_function(G), sf)
        t_val = r_h["cpu_s"]
        r_val = r_h["linear_relaxation"]
        h_nodes  = r_h.get("nodes_explored", None)
        h_nvars  = r_h.get("n_variables",    None)
        h_ncons  = r_h.get("n_constraints",  None)
        h_gap    = r_h.get("lp_gap",         None)

        # --- baseline aleatório ---
        rand_times, rand_relax  = [], []
        rand_nodes, rand_nvars  = [], []
        rand_ncons, rand_gaps   = [], []

        for perm in sample_permutations_fn(nodes, k_perms):
            r = solver_fn(G, perm, sf)
            rand_times.append(r["cpu_s"])
            rand_relax.append(r["linear_relaxation"])
            rand_nodes.append(r.get("nodes_explored", None))
            rand_nvars.append(r.get("n_variables",    None))
            rand_ncons.append(r.get("n_constraints",  None))
            rand_gaps.append( r.get("lp_gap",         None))
        
        print("time ", t_val)
        print("relax ", r_val)
        print("nodes ", h_nodes)
        print("nvars ", h_nvars)
        print("ncons ", h_ncons)
        print("gaps ", h_gap)

        
        print("rand_times", rand_times)
        print("rand_relax", rand_relax)
        print("rand_nodes", rand_nodes)
        print("rand_nvars", rand_nvars)
        print("rand_ncons", rand_ncons)
        print("rand_gaps", rand_gaps)


        
        
        
        

        # percentis principais
        perc_t = percentile_lower_is_better(rand_times, t_val)
        perc_r = percentile_lower_is_better(rand_relax, r_val)
        percentis_time.append(perc_t)
        percentis_relaxation.append(perc_r)

        # percentis estruturais (só calcula se os dados existem)
        perc_nodes = perc_nvars = perc_gap = None
        if h_nodes is not None and all(v is not None for v in rand_nodes):
            perc_nodes = percentile_lower_is_better(rand_nodes, h_nodes)
            percentis_nodes.append(perc_nodes)
        if h_nvars is not None and all(v is not None for v in rand_nvars):
            perc_nvars = percentile_lower_is_better(rand_nvars, h_nvars)
            percentis_nvars.append(perc_nvars)
        if h_gap is not None and all(v is not None for v in rand_gaps):
            perc_gap = percentile_lower_is_better(rand_gaps, h_gap)
            percentis_lp_gap.append(perc_gap)

        record = {
            "graph_index":           i,
            "n_nodes":               n,
            "n_edges":               G.number_of_edges(),
            "p":                     p,
            "stair_factor":          sf,
            # heurística
            "heuristic_time":        t_val,
            "heuristic_relaxation":  r_val,
            "heuristic_nodes":       h_nodes,
            "heuristic_nvars":       h_nvars,
            "heuristic_ncons":       h_ncons,
            "heuristic_lp_gap":      h_gap,
            # baseline
            "random_times":          rand_times,
            "random_relaxations":    rand_relax,
            "random_nodes":          rand_nodes,
            "random_nvars":          rand_nvars,
            "random_ncons":          rand_ncons,
            "random_lp_gaps":        rand_gaps,
            # percentis
            "percentile_time":       perc_t,
            "percentile_relaxation": perc_r,
            "percentile_nodes":      perc_nodes,
            "percentile_nvars":      perc_nvars,
            "percentile_lp_gap":     perc_gap,
        }
        records.append(record)
        print(
            f"  iter {i:03d} | "
            f"perc_time={perc_t:6.2f}%  "
            f"perc_relax={perc_r:6.2f}%  "
            f"perc_nodes={perc_nodes!s:>6}%  "
            f"perc_nvars={perc_nvars!s:>6}%  "
            f"perc_gap={perc_gap!s:>6}%"
        )

    def _mean_or_none(lst):
        lst = [v for v in lst if v is not None]
        return float(np.mean(lst)) if lst else None

    summary = {
        "experiment_tag":            experiment_tag or heuristic_function.__name__,
        "timestamp":                 datetime.now().isoformat(),
        "params":                    {"num_graphs": num_graphs, "n": n, "p": p, "k_perms": k_perms},
        "mean_percentile_time":      float(np.mean(percentis_time)),
        "mean_percentile_relax":     float(np.mean(percentis_relaxation)),
        "std_percentile_time":       float(np.std(percentis_time)),
        "std_percentile_relax":      float(np.std(percentis_relaxation)),
        "mean_percentile_nodes":     _mean_or_none(percentis_nodes),
        "mean_percentile_nvars":     _mean_or_none(percentis_nvars),
        "mean_percentile_lp_gap":    _mean_or_none(percentis_lp_gap),
        "pearson_corr":              float(pearsonr(percentis_time, percentis_relaxation)[0]),
        "spearman_corr":             float(spearmanr(percentis_time, percentis_relaxation)[0]),
        "records":                   records,
    }

    tag   = experiment_tag or heuristic_function.__name__
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = pathlib.Path(save_dir) / f"{tag}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResultados salvos em: {fname}")
    print(f"Percentil tempo médio:      {summary['mean_percentile_time']:.2f}%")
    print(f"Percentil relaxação médio:  {summary['mean_percentile_relax']:.2f}%")
    print(f"Percentil nós médio:        {summary['mean_percentile_nodes']}")
    print(f"Percentil nvars médio:      {summary['mean_percentile_nvars']}")
    print(f"Percentil lp_gap médio:     {summary['mean_percentile_lp_gap']}")
    print(f"Correlação Pearson:         {summary['pearson_corr']:.4f}")

    return summary

def load_results(filepath: str) -> dict:
    """Carrega um arquivo JSON salvo por run_experiment_v2."""
    with open(filepath) as f:
        return json.load(f)



if __name__ == "__main__":
    