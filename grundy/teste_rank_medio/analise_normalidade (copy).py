import random, itertools, statistics
from collections import defaultdict
from math import comb, factorial

Graph = dict

def make_graph(n):
    return {v: set() for v in range(n)}

def add_edge(G, u, v):
    G[u].add(v); G[v].add(u)

def erdos_renyi(n, p, seed):
    rng = random.Random(seed)
    G = make_graph(n)
    for u in range(n):
        for v in range(u+1, n):
            if rng.random() < p:
                add_edge(G, u, v)
    return G

def slo(G):
    n = len(G)
    degree = {v: len(G[v]) for v in G}
    removed = [False]*n; removal_order = []
    buckets = defaultdict(set)
    for v,d in degree.items(): buckets[d].add(v)
    alpha = 0
    for _ in range(n):
        for d in range(n):
            if buckets[d]:
                v = next(iter(buckets[d])); buckets[d].remove(v); break
        removed[v] = True; removal_order.append(v); alpha = max(alpha, degree[v])
        for u in G[v]:
            if not removed[u]:
                buckets[degree[u]].discard(u)
                degree[u] -= 1; buckets[degree[u]].add(u)
    sigma = [0]*n
    for rank, v in enumerate(reversed(removal_order)): sigma[v] = rank
    return sigma, alpha

def random_order(G, seed):
    n = len(G); perm = list(range(n))
    random.Random(seed).shuffle(perm)
    sigma = [0]*n
    for rank,v in enumerate(perm): sigma[v] = rank
    return sigma

def grundy_color(G, order):
    color = [-1]*len(G)
    for v in order:
        nc = {color[u] for u in G[v] if color[u]>=0}
        c = 0
        while c in nc: c += 1
        color[v] = c
    return color

# ---------------------------------------------------------------------------
# Estrutura de solução com multiplicidade
# ---------------------------------------------------------------------------

def coloring_key(col):
    """
    Chave canônica de uma coloração: tupla das classes ordenadas.
    Duas ordens que geram o mesmo particionamento têm a mesma chave.
    """
    classes = defaultdict(list)
    for v, c in enumerate(col):
        classes[c].append(v)
    # ordena os membros de cada classe e as classes pelo menor membro
    frozen = tuple(
        tuple(sorted(members))
        for members in sorted(classes.values(), key=lambda m: min(m))
    )
    return frozen

def all_grundy_with_multiplicity(G):
    """
    Enumera todas as permutações e retorna:
      - gamma: número de Grundy
      - colorings: dict {chave_canônica: {"coloring": [...], "multiplicity": int}}
      - n_orders: número total de ordens ótimas
    """
    n = len(G); best = 0
    colorings = {}   # chave → {coloring, multiplicity}
    n_orders  = 0

    for perm in itertools.permutations(range(n)):
        col = grundy_color(G, list(perm))
        k   = max(col) + 1
        if k > best:
            best = k
            colorings = {}
            n_orders  = 0
        if k == best:
            n_orders += 1
            key = coloring_key(col)
            if key not in colorings:
                colorings[key] = {"coloring": col, "multiplicity": 0}
            colorings[key]["multiplicity"] += 1

    return best, colorings, n_orders

# ---------------------------------------------------------------------------
# Rank médio — três variantes
# ---------------------------------------------------------------------------

def best_rank(gamma):
    return statistics.mean(range(gamma))

def rank_medio_por_ordem(colorings, n_orders, sigma):
    """
    Peso proporcional à multiplicidade de cada coloração.
    Equivalente ao cálculo original (todas as ordens com peso 1),
    mas sem precisar iterar n! pares (ordem, coloração).
    """
    weighted_ranks = []
    for info in colorings.values():
        col  = info["coloring"]
        mult = info["multiplicity"]
        classes = defaultdict(list)
        for v, c in enumerate(col): classes[c].append(v)
        rep_rank = sum(
            sigma[min(members, key=lambda v: sigma[v])]
            for members in classes.values()
        ) / len(classes)
        weighted_ranks.extend([rep_rank] * mult)   # peso = multiplicidade

    gamma = max(max(info["coloring"]) for info in colorings.values()) + 1
    return statistics.mean(weighted_ranks) - best_rank(gamma)

def rank_medio_por_coloracao(colorings, sigma):
    """
    Cada coloração distinta tem peso 1, independentemente da multiplicidade.
    Responde: 'Dentre as soluções estruturalmente distintas, qual o
    rank médio do representante canônico?'
    """
    ranks = []
    gamma = None
    for info in colorings.values():
        col = info["coloring"]
        if gamma is None: gamma = max(col) + 1
        classes = defaultdict(list)
        for v, c in enumerate(col): classes[c].append(v)
        for members in classes.values():
            ranks.append(sigma[min(members, key=lambda v: sigma[v])])
    return statistics.mean(ranks) - best_rank(gamma)

def rank_medio_ponderado_log(colorings, sigma):
    """
    Peso logarítmico: log(multiplicidade + 1).
    Compromisso entre as duas métricas anteriores — reduz o impacto
    de colorações muito acessíveis sem ignorar a multiplicidade.
    """
    weighted_ranks = []
    weights        = []
    gamma = None
    for info in colorings.values():
        col  = info["coloring"]
        mult = info["multiplicity"]
        if gamma is None: gamma = max(col) + 1
        w = math.log(mult + 1)
        classes = defaultdict(list)
        for v, c in enumerate(col): classes[c].append(v)
        rep_rank = sum(
            sigma[min(members, key=lambda v: sigma[v])]
            for members in classes.values()
        ) / len(classes)
        weighted_ranks.append(rep_rank * w)
        weights.append(w)
    mean_r = sum(weighted_ranks) / sum(weights)
    return mean_r - best_rank(gamma)

# ---------------------------------------------------------------------------
# Análise de multiplicidade
# ---------------------------------------------------------------------------

def multiplicity_stats(colorings, n_orders):
    """
    Retorna estatísticas sobre a distribuição de multiplicidades.
    """
    mults = [info["multiplicity"] for info in colorings.values()]
    n_distinct = len(mults)
    n_total    = sum(mults)
    assert n_total == n_orders

    return {
        "n_distinct":    n_distinct,
        "n_orders":      n_orders,
        "ratio":         n_distinct / n_orders,   # próximo de 0 = muita redundância
        "mult_max":      max(mults),
        "mult_min":      min(mults),
        "mult_mean":     statistics.mean(mults),
        "mult_median":   statistics.median(mults),
        "concentration": max(mults) / n_orders,   # fração das ordens na coloração mais acessível
    }

def slo_multiplicity_rank(colorings, sigma):
    """
    Para a coloração que a SLO 'escolheria' (aquela cujo representante
    canônico tem menor rank sob sigma), retorna sua multiplicidade.
    Permite verificar se a SLO tende a selecionar colorações acessíveis.
    """
    best_key  = None
    best_rank_val = float("inf")
    for key, info in colorings.items():
        col = info["coloring"]
        classes = defaultdict(list)
        for v, c in enumerate(col): classes[c].append(v)
        r = sum(sigma[min(m, key=lambda v: sigma[v])] for m in classes.values())
        if r < best_rank_val:
            best_rank_val = r
            best_key = key
    return colorings[best_key]["multiplicity"]

import math

# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

size   = 8
configs = [(size,0.1,42),(size,0.25,59),(size,0.5,76),(size,0.75,93),(size,0.9,110)]
N_RAND = 500

print("=" * 100)
print("  ANÁLISE DE MULTIPLICIDADE E RANK MÉDIO")
print("=" * 100)

for n, p, seed in configs:
    G = erdos_renyi(n, p, seed)
    gamma, colorings, n_orders = all_grundy_with_multiplicity(G)
    ms = multiplicity_stats(colorings, n_orders)

    print(f"\n{'─'*100}")
    print(f"  n={n}, p={p}  |  Γ={gamma}  |  "
          f"colorações distintas={ms['n_distinct']}  |  "
          f"ordens ótimas={ms['n_orders']}  |  "
          f"ratio={ms['ratio']:.4f}")
    print(f"  multiplicidade: min={ms['mult_min']}  max={ms['mult_max']}  "
          f"média={ms['mult_mean']:.1f}  mediana={ms['mult_median']}  "
          f"concentração={ms['concentration']:.3f}")
    print(f"{'─'*100}")

    sg, _ = slo(G)
    slo_mult = slo_multiplicity_rank(colorings, sg)

    # rank médio SLO nas três variantes
    r_slo_ord = rank_medio_por_ordem(colorings, n_orders, sg)
    r_slo_col = rank_medio_por_coloracao(colorings, sg)
    r_slo_log = rank_medio_ponderado_log(colorings, sg)

    # distribuição das três métricas para N_RAND aleatórias
    r_ord_rnds, r_col_rnds, r_log_rnds, mult_rnds = [], [], [], []
    for k in range(N_RAND):
        sr = random_order(G, seed=seed + k*13 + 2000)
        r_ord_rnds.append(rank_medio_por_ordem(colorings, n_orders, sr))
        r_col_rnds.append(rank_medio_por_coloracao(colorings, sr))
        r_log_rnds.append(rank_medio_ponderado_log(colorings, sr))
        mult_rnds.append(slo_multiplicity_rank(colorings, sr))

    def pval(r_slo, rnds): return sum(1 for r in rnds if r <= r_slo)/len(rnds)
    def z(r_slo, rnds):
        mu = statistics.mean(rnds); sd = statistics.stdev(rnds)
        return (mu - r_slo)/sd if sd > 0 else 0.0
    def sig(p): return "**" if p<0.01 else ("*" if p<0.05 else "ns")

    print(f"  {'Métrica':<30} {'r_SLO':>8} {'μ_rnd':>8} {'z':>7} {'p-val':>7} {'sig':>4}")
    print(f"  {'─'*65}")

    for label, r_slo_v, rnds in [
        ("rank por ordem (peso=mult)",    r_slo_ord, r_ord_rnds),
        ("rank por coloração (peso=1)",   r_slo_col, r_col_rnds),
        ("rank ponderado log(mult+1)",    r_slo_log, r_log_rnds),
    ]:
        mu  = statistics.mean(rnds)
        pv  = pval(r_slo_v, rnds)
        zv  = z(r_slo_v, rnds)
        print(f"  {label:<30} {r_slo_v:>8.3f} {mu:>8.3f} {zv:>7.3f} {pv:>7.4f} {sig(pv):>4}")

    # multiplicidade da coloração escolhida pela SLO vs aleatórias
    mu_mult = statistics.mean(mult_rnds)
    sd_mult = statistics.stdev(mult_rnds)
    pv_mult = sum(1 for m in mult_rnds if m >= slo_mult)/N_RAND  # maior é melhor
    zv_mult = (slo_mult - mu_mult)/sd_mult if sd_mult > 0 else 0.0
    print(f"  {'mult. da coloração escolhida':<30} {slo_mult:>8} {mu_mult:>8.1f} "
          f"{zv_mult:>7.3f} {pv_mult:>7.4f} {sig(pv_mult):>4}")
    print(f"  (positivo = SLO escolhe coloração mais acessível que o acaso)")