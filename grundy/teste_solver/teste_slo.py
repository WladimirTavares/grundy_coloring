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


# ---------------------------------------------------------------------------
# Experimento com persistência
# ---------------------------------------------------------------------------

def run_experiment_v2(
    heuristic_function,
    num_graphs: int = 50,
    n: int = 50,
    p: float = 0.5,
    k_perms: int = 200,
    save_dir: str = "results",
    experiment_tag: str = "",
    # injete suas funções aqui:
    solver_fn=None,
    stair_factor_fn=None,
    sample_permutations_fn=None,
):
    """
    Versão estendida de run_experiment que persiste todos os dados brutos.

    Parâmetros
    ----------
    heuristic_function : callable
        Função que recebe um grafo e devolve uma ordem (lista de nós).
    num_graphs : int
        Número de grafos Erdős–Rényi amostrados.
    n, p : int, float
        Parâmetros do modelo G(n, p).
    k_perms : int
        Número de permutações aleatórias por grafo.
    save_dir : str
        Diretório onde os resultados serão salvos.
    experiment_tag : str
        Rótulo livre para identificar o experimento (nome da heurística, etc.).
    solver_fn : callable
        Referência para solver_carvalho_representante3.
    stair_factor_fn : callable
        Referência para stair_factor.
    sample_permutations_fn : callable
        Referência para sample_permutations.

    Retorna
    -------
    dict com todos os dados brutos e estatísticas agregadas,
    e salva em  <save_dir>/<tag>_<timestamp>.json
    """
    import networkx as nx

    assert solver_fn is not None, "Passe solver_fn=solver_carvalho_representante3"
    assert stair_factor_fn is not None, "Passe stair_factor_fn=stair_factor"
    assert sample_permutations_fn is not None, "Passe sample_permutations_fn=sample_permutations"

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    records = []          # um registro por grafo
    percentis_time = []
    percentis_relaxation = []

    for i in range(num_graphs):
        G = nx.erdos_renyi_graph(n, p)
        nodes = list(G.nodes())
        sf = stair_factor_fn(G)

        # --- heurística ---
        r_h = solver_fn(G, heuristic_function(G), sf)
        t_val = r_h["cpu_s"]
        r_val = r_h["linear_relaxation"]

        # --- baseline aleatório ---
        rand_times, rand_relax = [], []
        for perm in sample_permutations_fn(nodes, k_perms):
            r = solver_fn(G, perm, sf)
            rand_times.append(r["cpu_s"])
            rand_relax.append(r["linear_relaxation"])

        perc_t = percentile_lower_is_better(rand_times, t_val)
        perc_r = percentile_higher_is_better(rand_relax, r_val)
        percentis_time.append(perc_t)
        percentis_relaxation.append(perc_r)

        record = {
            "graph_index":            i,
            "n_nodes":                n,
            "n_edges":                G.number_of_edges(),
            "p":                      p,
            "stair_factor":           sf,
            "heuristic_time":         t_val,
            "heuristic_relaxation":   r_val,
            "random_times":           rand_times,
            "random_relaxations":     rand_relax,
            "percentile_time":        perc_t,
            "percentile_relaxation":  perc_r,
        }
        records.append(record)
        print(f"  iter {i:03d} | perc_time={perc_t:6.2f}%  perc_relax={perc_r:6.2f}%")

    # --- resumo ---
    summary = {
        "experiment_tag":           experiment_tag or heuristic_function.__name__,
        "timestamp":                datetime.now().isoformat(),
        "params":                   {"num_graphs": num_graphs, "n": n, "p": p, "k_perms": k_perms},
        "mean_percentile_time":     float(np.mean(percentis_time)),
        "mean_percentile_relax":    float(np.mean(percentis_relaxation)),
        "std_percentile_time":      float(np.std(percentis_time)),
        "std_percentile_relax":     float(np.std(percentis_relaxation)),
        "pearson_corr":             float(pearsonr(percentis_time, percentis_relaxation)[0]),
        "spearman_corr":            float(spearmanr(percentis_time, percentis_relaxation)[0]),
        "records":                  records,
    }

    tag = experiment_tag or heuristic_function.__name__
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = pathlib.Path(save_dir) / f"{tag}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResultados salvos em: {fname}")
    print(f"Percentil tempo médio:      {summary['mean_percentile_time']:.2f}%")
    print(f"Percentil relaxação médio:  {summary['mean_percentile_relax']:.2f}%")
    print(f"Correlação Pearson:         {summary['pearson_corr']:.4f}")

    return summary


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


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def plot_all_visualizations(summary: dict, save_dir: str = "plots", show: bool = True):
    """
    Gera as 5 visualizações a partir de um dict retornado por run_experiment_v2
    (ou carregado via load_results).

    Salva cada figura em <save_dir>/ e exibe se show=True.
    """
    _apply_dark_style()
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    records = summary["records"]
    perc_t  = [r["percentile_time"]       for r in records]
    perc_r  = [r["percentile_relaxation"] for r in records]
    tag     = summary.get("experiment_tag", "experiment")

    # escolhe um grafo representativo (mediano em perc_t)
    med_idx = int(np.argsort(perc_t)[len(perc_t) // 2])
    rep_rec = records[med_idx]

    _plot1_scatter(perc_t, perc_r, tag, save_dir)
    _plot2_histograms(perc_t, perc_r, tag, save_dir)
    _plot3_boxplots(records, tag, save_dir)
    _plot4_ecdf(rep_rec, tag, save_dir)
    _plot5_rolling_corr(perc_t, perc_r, tag, save_dir)

    if show:
        plt.show()


# --- Visualização 1: Scatter percentil tempo × relaxação ------------------

def _plot1_scatter(perc_t, perc_r, tag, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.axvline(50, color=PALETTE["grid"], lw=1.2, zorder=1)
    ax.axhline(50, color=PALETTE["grid"], lw=1.2, zorder=1)

    sc = ax.scatter(
        perc_t, perc_r,
        c=np.array(perc_t) - np.array(perc_r),   # cor = ganho relativo
        cmap="RdYlGn", vmin=-80, vmax=80,
        s=70, alpha=0.85, edgecolors="none", zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Perc(tempo) − Perc(relaxação)", color=PALETTE["text_muted"], fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text_muted"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text_muted"])

    # quadrante de interesse
    ax.fill_betweenx([50, 100], [50, 50], [100, 100],
                     color=PALETTE["heuristic"], alpha=0.06, zorder=0)
    ax.text(97, 97, "Tempo ↓\nRelax ~", ha="right", va="top",
            fontsize=8, color=PALETTE["accent"], alpha=0.8)

    pearson = pearsonr(perc_t, perc_r)[0]
    ax.set_title(
        f"[{tag}]  Percentil Tempo × Percentil Relaxação\n"
        f"Pearson r = {pearson:.3f}  |  n = {len(perc_t)} grafos",
        pad=12,
    )
    ax.set_xlabel("Percentil de Tempo  (maior = mais rápido que aleatório)")
    ax.set_ylabel("Percentil de Relaxação  (maior = relaxação melhor)")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.grid(True, zorder=0)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/1_scatter_perc_tempo_relax.png", dpi=150,
                facecolor=PALETTE["bg"])
    print("  ✓ plot 1 salvo")


# --- Visualização 2: Histogramas lado a lado ------------------------------

def _plot2_histograms(perc_t, perc_r, tag, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(f"[{tag}]  Distribuição dos Percentis por Grafo", fontsize=14, y=1.02)

    data_pairs = [
        (axes[0], perc_t, PALETTE["heuristic"],
         "Percentil de Tempo\n(heurística vs. aleatório)",
         "% de permutações aleatórias mais lentas"),
        (axes[1], perc_r, PALETTE["random"],
         "Percentil de Relaxação\n(heurística vs. aleatório)",
         "% de permutações aleatórias com relaxação pior"),
    ]

    for ax, data, color, title, xlabel in data_pairs:
        bins = np.linspace(0, 100, 21)
        ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor=PALETTE["bg"], zorder=3)
        mean_val = np.mean(data)
        ax.axvline(mean_val, color=PALETTE["accent"], lw=2, ls="--", zorder=4,
                   label=f"Média = {mean_val:.1f}%")
        ax.axvline(50, color=PALETTE["text_muted"], lw=1.2, ls=":", zorder=4,
                   label="Referência = 50%")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Nº de Grafos")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", zorder=0)
        ax.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/2_histogramas_percentis.png", dpi=150,
                facecolor=PALETTE["bg"])
    print("  ✓ plot 2 salvo")


# --- Visualização 3: Box plots (tempo e relaxação reais) ------------------

def _plot3_boxplots(records, tag, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(f"[{tag}]  Distribuições Reais: Heurística vs. Aleatório", fontsize=14, y=1.01)

    # ---- painel de tempo ----
    ax = axes[0]
    all_rand_times = [t for r in records for t in r["random_times"]]
    heur_times     = [r["heuristic_time"] for r in records]

    bp = ax.boxplot(
        [all_rand_times, heur_times],
        labels=["Aleatório", "Heurística"],
        patch_artist=True,
        medianprops=dict(color=PALETTE["accent"], lw=2),
        whiskerprops=dict(color=PALETTE["text_muted"]),
        capprops=dict(color=PALETTE["text_muted"]),
        flierprops=dict(marker="o", color=PALETTE["text_muted"], alpha=0.3, markersize=3),
    )
    for patch, color in zip(bp["boxes"], [PALETTE["random"], PALETTE["heuristic"]]):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.set_title("Tempo de Resolução (s)")
    ax.set_ylabel("CPU (segundos)")
    ax.grid(True, axis="y")

    # ---- painel de relaxação ----
    ax = axes[1]
    all_rand_relax = [r2 for r in records for r2 in r["random_relaxations"]]
    heur_relax     = [r["heuristic_relaxation"] for r in records]

    bp2 = ax.boxplot(
        [all_rand_relax, heur_relax],
        labels=["Aleatório", "Heurística"],
        patch_artist=True,
        medianprops=dict(color=PALETTE["accent"], lw=2),
        whiskerprops=dict(color=PALETTE["text_muted"]),
        capprops=dict(color=PALETTE["text_muted"]),
        flierprops=dict(marker="o", color=PALETTE["text_muted"], alpha=0.3, markersize=3),
    )
    for patch, color in zip(bp2["boxes"], [PALETTE["random"], PALETTE["heuristic"]]):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    ax.set_title("Valor da Relaxação Linear")
    ax.set_ylabel("Objective value (relaxação)")
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(f"{save_dir}/3_boxplots_tempo_relax.png", dpi=150,
                facecolor=PALETTE["bg"])
    print("  ✓ plot 3 salvo")


# --- Visualização 4: ECDF de grafo representativo -------------------------

def _plot4_ecdf(rep_rec, tag, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        f"[{tag}]  ECDF — Grafo representativo #{rep_rec['graph_index']}\n"
        f"({rep_rec['n_nodes']} vértices, {rep_rec['n_edges']} arestas)",
        fontsize=13, y=1.02,
    )

    panels = [
        (axes[0],
         rep_rec["random_times"],
         rep_rec["heuristic_time"],
         "Tempo de Resolução (s)",
         "menor é melhor",
         PALETTE["heuristic"],
         True,   # ← marcar à esquerda é bom
         ),
        (axes[1],
         rep_rec["random_relaxations"],
         rep_rec["heuristic_relaxation"],
         "Valor da Relaxação Linear",
         "maior é melhor",
         PALETTE["random"],
         False,  # ← marcar à direita é bom
         ),
    ]

    for ax, rand_vals, heur_val, xlabel, subtitle, hcolor, left_is_good in panels:
        sorted_vals = np.sort(rand_vals)
        ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        ax.plot(sorted_vals, ecdf_y, color=PALETTE["random"], lw=2,
                label="Distribuição aleatória")
        ax.fill_between(sorted_vals, ecdf_y, alpha=0.15, color=PALETTE["random"])

        perc = np.mean(np.array(rand_vals) <= heur_val) * 100
        ax.axvline(heur_val, color=hcolor, lw=2.5, ls="--",
                   label=f"Heurística\n(percentil {perc:.1f}%)")

        # sombra da região "melhor que a heurística"
        if left_is_good:
            mask = sorted_vals <= heur_val
            ax.fill_between(sorted_vals[mask], ecdf_y[mask], 0,
                            color=hcolor, alpha=0.18, label="Região melhor")
        else:
            mask = sorted_vals >= heur_val
            ax.fill_between(sorted_vals[mask], ecdf_y[mask], 1,
                            color=hcolor, alpha=0.18, label="Região melhor")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probabilidade acumulada")
        ax.set_title(f"{xlabel}\n({subtitle})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/4_ecdf_grafo_representativo.png", dpi=150,
                facecolor=PALETTE["bg"])
    print("  ✓ plot 4 salvo")


# --- Visualização 5: Correlação deslizante --------------------------------

def _plot5_rolling_corr(perc_t, perc_r, tag, save_dir):
    window = max(5, len(perc_t) // 8)
    rolling = _rolling_corr(perc_t, perc_r, window)
    x_vals  = list(range(len(rolling)))
    valid   = [(i, v) for i, v in enumerate(rolling) if v is not None]
    xi, yi  = zip(*valid)

    cum_corr = []
    for i in range(1, len(perc_t) + 1):
        if i < 2:
            cum_corr.append(None)
        elif np.std(perc_t[:i]) == 0 or np.std(perc_r[:i]) == 0:
            cum_corr.append(0.0)
        else:
            cum_corr.append(pearsonr(perc_t[:i], perc_r[:i])[0])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        f"[{tag}]  Ausência de Correlação: Percentil Tempo vs. Relaxação\n"
        f"Correlação global: Pearson = {pearsonr(perc_t, perc_r)[0]:.4f} | "
        f"Spearman = {spearmanr(perc_t, perc_r)[0]:.4f}",
        fontsize=13, y=1.01,
    )

    # painel superior: scatter com linha de regressão
    ax0 = axes[0]
    ax0.scatter(perc_t, perc_r, color=PALETTE["heuristic"], alpha=0.6,
                s=55, edgecolors="none", label="Grafos")
    m, b = np.polyfit(perc_t, perc_r, 1)
    xs = np.linspace(0, 100, 200)
    ax0.plot(xs, m * xs + b, color=PALETTE["accent"], lw=2, ls="--",
             label=f"Regressão: y={m:.2f}x+{b:.1f}")
    ax0.axhline(np.mean(perc_r), color=PALETTE["random"], lw=1.2, ls=":",
                label=f"Média relax = {np.mean(perc_r):.1f}%")
    ax0.set_xlabel("Percentil Tempo"); ax0.set_ylabel("Percentil Relaxação")
    ax0.set_xlim(0, 100); ax0.set_ylim(0, 100)
    ax0.legend(fontsize=9); ax0.grid(True)

    # painel inferior: correlação acumulada
    ax1 = axes[1]
    valid_cum = [(i, v) for i, v in enumerate(cum_corr) if v is not None]
    if valid_cum:
        xc, yc = zip(*valid_cum)
        ax1.plot(xc, yc, color=PALETTE["random"], lw=2, label="Pearson acumulado")
        ax1.fill_between(xc, yc, 0, where=np.array(yc) > 0,
                         color=PALETTE["random"], alpha=0.15)
        ax1.fill_between(xc, yc, 0, where=np.array(yc) < 0,
                         color=PALETTE["heuristic"], alpha=0.15)
    ax1.axhline(0,  color=PALETTE["accent"],     lw=1.5, ls="--", label="r = 0")
    ax1.axhline(0.3,  color=PALETTE["text_muted"], lw=0.8, ls=":", label="|r| = 0.3")
    ax1.axhline(-0.3, color=PALETTE["text_muted"], lw=0.8, ls=":")
    ax1.set_xlabel("Número de grafos processados (cumulativo)")
    ax1.set_ylabel("Correlação de Pearson")
    ax1.set_title(f"Correlação acumulada converge para ~0 → independência estatística", fontsize=11)
    ax1.set_ylim(-1, 1); ax1.legend(fontsize=9); ax1.grid(True)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/5_correlacao_deslizante.png", dpi=150,
                facecolor=PALETTE["bg"])
    print("  ✓ plot 5 salvo")


# ---------------------------------------------------------------------------
# Visualizações focadas (3 plots recomendados, tempo e relaxação separados)
# ---------------------------------------------------------------------------

def _ecdf_panel(ax, rand_vals, heur_val, xlabel, left_is_good, color, tag_label):
    """Desenha um painel ECDF completo em um eixo existente."""
    sorted_vals = np.sort(rand_vals)
    ecdf_y      = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    ax.step(sorted_vals, ecdf_y, color=PALETTE["random"], lw=2,
            where="post", label="Permutações aleatórias", zorder=3)
    ax.fill_between(sorted_vals, ecdf_y, alpha=0.12,
                    color=PALETTE["random"], step="post", zorder=2)

    perc = np.mean(np.array(rand_vals) <= heur_val) * 100
    ax.axvline(heur_val, color=color, lw=2.5, ls="--", zorder=4,
               label=f"Heurística  (perc. {perc:.1f}%)")

    if left_is_good:
        mask = sorted_vals <= heur_val
        if mask.any():
            ax.fill_between(sorted_vals[mask], ecdf_y[mask], 0,
                            color=color, alpha=0.22, step="post",
                            label="Aleatórias mais lentas", zorder=1)
    else:
        mask = sorted_vals >= heur_val
        if mask.any():
            ax.fill_between(sorted_vals[mask], ecdf_y[mask], 1,
                            color=color, alpha=0.22, step="post",
                            label="Aleatórias com relax. pior", zorder=1)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Prob. acumulada", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True)


def plot_focused(filepath: str, save_dir: str = "plots", show: bool = True):
    """
    Gera os 3 plots recomendados para o setup restrito, com tempo e relaxação
    em figuras completamente separadas.

    Plots produzidos
    ----------------
    A) ECDF — Tempo          : uma figura só com o painel de tempo,
                               usando todos os grafos (painel por grafo +
                               painel agregado).
    B) ECDF — Relaxação      : idem para a relaxação linear.
    C) Scatter + Histogramas : figura com 3 painéis:
                               scatter percentil-tempo × percentil-relaxação,
                               histograma de percentis de tempo,
                               histograma de percentis de relaxação.

    Parâmetros
    ----------
    filepath : str
        Caminho para o JSON salvo por run_experiment_v2.
    save_dir : str
        Diretório de saída para os PNGs.
    show : bool
        Se True, chama plt.show() ao final.
    """
    summary = load_results(filepath)
    _apply_dark_style()
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    records = summary["records"]
    tag     = summary.get("experiment_tag", "experiment")
    params  = summary.get("params", {})
    n_nodes = params.get("n", "?")
    k_perms = params.get("k_perms", "?")
    subtitle = f"n={n_nodes} vértices | k={k_perms} permutações | {len(records)} grafos"

    perc_t = [r["percentile_time"]       for r in records]
    perc_r = [r["percentile_relaxation"] for r in records]

    # escolhe grafo representativo (mediano no percentil de tempo)
    med_idx  = int(np.argsort(perc_t)[len(perc_t) // 2])
    rep_rec  = records[med_idx]

    # ------------------------------------------------------------------ #
    # Plot A — ECDF Tempo                                                  #
    # ------------------------------------------------------------------ #
    n_rec   = len(records)
    n_cols  = min(4, n_rec)
    n_rows  = math.ceil(n_rec / n_cols) + 1   # +1 para painel agregado

    fig_t   = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    fig_t.patch.set_facecolor(PALETTE["bg"])
    fig_t.suptitle(
        f"[{tag}]  ECDF — Tempo de Resolução\n{subtitle}",
        fontsize=13, y=1.01,
    )
    gs_t = gridspec.GridSpec(n_rows, n_cols, figure=fig_t,
                             hspace=0.55, wspace=0.35)

    # painéis individuais por grafo
    for idx, rec in enumerate(records):
        row, col = divmod(idx, n_cols)
        ax = fig_t.add_subplot(gs_t[row, col])
        _ecdf_panel(
            ax, rec["random_times"], rec["heuristic_time"],
            "Tempo (s)", left_is_good=True,
            color=PALETTE["heuristic"],
            tag_label=tag,
        )
        ax.set_title(f"Grafo {rec['graph_index']}  "
                     f"(perc={rec['percentile_time']:.0f}%)",
                     fontsize=9)
        ax.set_ylabel("")
        if col != 0:
            ax.set_ylabel("")

    # painel agregado (última linha, largura total)
    ax_agg = fig_t.add_subplot(gs_t[n_rows - 1, :])
    all_rand_t = [t for r in records for t in r["random_times"]]
    all_heur_t = [r["heuristic_time"] for r in records]
    # junta todos os valores de heurística como uma "distribuição"
    sorted_rand = np.sort(all_rand_t)
    ecdf_rand   = np.arange(1, len(sorted_rand) + 1) / len(sorted_rand)
    ax_agg.step(sorted_rand, ecdf_rand, color=PALETTE["random"], lw=2,
                where="post", label="Aleatórias (todos os grafos)", zorder=3)
    ax_agg.fill_between(sorted_rand, ecdf_rand, alpha=0.12,
                        color=PALETTE["random"], step="post")
    for ht in all_heur_t:
        ax_agg.axvline(ht, color=PALETTE["heuristic"], lw=0.8,
                       alpha=0.5, ls="--", zorder=4)
    # linha fantasma para a legenda
    ax_agg.axvline(np.median(all_heur_t), color=PALETTE["heuristic"],
                   lw=2, ls="--", zorder=5,
                   label=f"Heurística (cada linha = 1 grafo)\n"
                         f"mediana = {np.median(all_heur_t):.3f}s")
    ax_agg.set_title("Agregado — todas as instâncias", fontsize=11)
    ax_agg.set_xlabel("Tempo de Resolução (s)")
    ax_agg.set_ylabel("Prob. acumulada")
    ax_agg.set_ylim(0, 1.05)
    ax_agg.legend(fontsize=9)
    ax_agg.grid(True)

    fname_t = f"{save_dir}/A_ecdf_tempo.png"
    fig_t.savefig(fname_t, dpi=150, facecolor=PALETTE["bg"],
                  bbox_inches="tight")
    print(f"  ✓ Plot A salvo → {fname_t}")

    # ------------------------------------------------------------------ #
    # Plot B — ECDF Relaxação                                              #
    # ------------------------------------------------------------------ #
    fig_r   = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    fig_r.patch.set_facecolor(PALETTE["bg"])
    fig_r.suptitle(
        f"[{tag}]  ECDF — Relaxação Linear\n{subtitle}",
        fontsize=13, y=1.01,
    )
    gs_r = gridspec.GridSpec(n_rows, n_cols, figure=fig_r,
                             hspace=0.55, wspace=0.35)

    for idx, rec in enumerate(records):
        row, col = divmod(idx, n_cols)
        ax = fig_r.add_subplot(gs_r[row, col])
        _ecdf_panel(
            ax, rec["random_relaxations"], rec["heuristic_relaxation"],
            "Relaxação", left_is_good=False,
            color=PALETTE["random"],
            tag_label=tag,
        )
        ax.set_title(f"Grafo {rec['graph_index']}  "
                     f"(perc={rec['percentile_relaxation']:.0f}%)",
                     fontsize=9)
        if col != 0:
            ax.set_ylabel("")

    ax_agg_r = fig_r.add_subplot(gs_r[n_rows - 1, :])
    all_rand_r = [v for r in records for v in r["random_relaxations"]]
    all_heur_r = [r["heuristic_relaxation"] for r in records]
    sorted_rand_r = np.sort(all_rand_r)
    ecdf_rand_r   = np.arange(1, len(sorted_rand_r) + 1) / len(sorted_rand_r)
    ax_agg_r.step(sorted_rand_r, ecdf_rand_r, color=PALETTE["random"],
                  lw=2, where="post", label="Aleatórias (todos os grafos)")
    ax_agg_r.fill_between(sorted_rand_r, ecdf_rand_r, alpha=0.12,
                          color=PALETTE["random"], step="post")
    for hr in all_heur_r:
        ax_agg_r.axvline(hr, color=PALETTE["accent"], lw=0.8,
                         alpha=0.5, ls="--")
    ax_agg_r.axvline(np.median(all_heur_r), color=PALETTE["accent"],
                     lw=2, ls="--",
                     label=f"Heurística (cada linha = 1 grafo)\n"
                           f"mediana = {np.median(all_heur_r):.3f}")
    ax_agg_r.set_title("Agregado — todas as instâncias", fontsize=11)
    ax_agg_r.set_xlabel("Valor da Relaxação Linear")
    ax_agg_r.set_ylabel("Prob. acumulada")
    ax_agg_r.set_ylim(0, 1.05)
    ax_agg_r.legend(fontsize=9)
    ax_agg_r.grid(True)

    fname_r = f"{save_dir}/B_ecdf_relaxacao.png"
    fig_r.savefig(fname_r, dpi=150, facecolor=PALETTE["bg"],
                  bbox_inches="tight")
    print(f"  ✓ Plot B salvo → {fname_r}")

    # ------------------------------------------------------------------ #
    # Plot C — Scatter + Histogramas                                       #
    # ------------------------------------------------------------------ #
    fig_c = plt.figure(figsize=(15, 5))
    fig_c.patch.set_facecolor(PALETTE["bg"])
    fig_c.suptitle(
        f"[{tag}]  Resumo dos Percentis\n{subtitle}",
        fontsize=13, y=1.02,
    )
    gs_c = gridspec.GridSpec(1, 3, figure=fig_c, wspace=0.35)

    # --- scatter ---
    ax_sc = fig_c.add_subplot(gs_c[0, 0])
    pearson_r = pearsonr(perc_t, perc_r)[0]
    sc = ax_sc.scatter(
        perc_t, perc_r,
        c=np.array(perc_t) - np.array(perc_r),
        cmap="RdYlGn", vmin=-80, vmax=80,
        s=80, alpha=0.85, edgecolors="none", zorder=3,
    )
    fig_c.colorbar(sc, ax=ax_sc, pad=0.02, label="Perc(t) − Perc(r)")
    ax_sc.axvline(50, color=PALETTE["grid"], lw=1, zorder=1)
    ax_sc.axhline(50, color=PALETTE["grid"], lw=1, zorder=1)
    ax_sc.fill_betweenx([50, 100], [50, 50], [100, 100],
                        color=PALETTE["heuristic"], alpha=0.06, zorder=0)
    m, b = np.polyfit(perc_t, perc_r, 1)
    xs = np.linspace(0, 100, 200)
    ax_sc.plot(xs, m * xs + b, color=PALETTE["accent"],
               lw=1.5, ls="--", label=f"r={pearson_r:.2f}")
    ax_sc.set_xlim(0, 100); ax_sc.set_ylim(0, 100)
    ax_sc.set_xlabel("Percentil Tempo\n(↑ heurística mais rápida)")
    ax_sc.set_ylabel("Percentil Relaxação\n(↑ heurística com relax. melhor)")
    ax_sc.set_title("Dissociação Tempo × Relaxação", fontsize=11)
    ax_sc.legend(fontsize=9); ax_sc.grid(True)

    # --- histograma tempo ---
    ax_ht = fig_c.add_subplot(gs_c[0, 1])
    n_bins = max(5, len(perc_t) // 4)
    bins   = np.linspace(0, 100, n_bins + 1)
    ax_ht.hist(perc_t, bins=bins, color=PALETTE["heuristic"],
               alpha=0.85, edgecolor=PALETTE["bg"], zorder=3)
    mean_t = np.mean(perc_t)
    ax_ht.axvline(mean_t, color=PALETTE["accent"], lw=2, ls="--",
                  label=f"Média = {mean_t:.1f}%")
    ax_ht.axvline(50, color=PALETTE["text_muted"], lw=1.2, ls=":",
                  label="Referência 50%")
    ax_ht.set_xlabel("Percentil de Tempo por Grafo")
    ax_ht.set_ylabel("Nº de Grafos")
    ax_ht.set_title("Distribuição — Percentil Tempo", fontsize=11)
    ax_ht.set_xlim(0, 100)
    ax_ht.legend(fontsize=9); ax_ht.grid(True, axis="y")

    # --- histograma relaxação ---
    ax_hr = fig_c.add_subplot(gs_c[0, 2])
    ax_hr.hist(perc_r, bins=bins, color=PALETTE["random"],
               alpha=0.85, edgecolor=PALETTE["bg"], zorder=3)
    mean_r = np.mean(perc_r)
    ax_hr.axvline(mean_r, color=PALETTE["accent"], lw=2, ls="--",
                  label=f"Média = {mean_r:.1f}%")
    ax_hr.axvline(50, color=PALETTE["text_muted"], lw=1.2, ls=":",
                  label="Referência 50%")
    ax_hr.set_xlabel("Percentil de Relaxação por Grafo")
    ax_hr.set_ylabel("Nº de Grafos")
    ax_hr.set_title("Distribuição — Percentil Relaxação", fontsize=11)
    ax_hr.set_xlim(0, 100)
    ax_hr.legend(fontsize=9); ax_hr.grid(True, axis="y")

    fname_c = f"{save_dir}/C_scatter_histogramas.png"
    fig_c.savefig(fname_c, dpi=150, facecolor=PALETTE["bg"],
                  bbox_inches="tight")
    print(f"  ✓ Plot C salvo → {fname_c}")

    if show:
        plt.show()

    return {"plot_A": fname_t, "plot_B": fname_r, "plot_C": fname_c}



if __name__ == "__main__":
    