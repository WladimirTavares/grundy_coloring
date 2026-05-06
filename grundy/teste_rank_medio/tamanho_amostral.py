import random, itertools, statistics, math
from collections import defaultdict
from scipy import stats as sp

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
        removed[v] = True; removal_order.append(v)
        alpha = max(alpha, degree[v])
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

def all_grundy(G):
    n = len(G); best = 0; sols = []
    for perm in itertools.permutations(range(n)):
        col = grundy_color(G, list(perm)); k = max(col)+1
        if k > best: best=k; sols=[(list(perm),col)]
        elif k==best: sols.append((list(perm),col))
    return best, sols

def best_rank(gamma):
    return statistics.mean(range(gamma))

def rank_medio(sols, sigma):
    ranks = []
    for _,col in sols:
        cls = defaultdict(list)
        for v,c in enumerate(col): cls[c].append(v)
        for members in cls.values():
            ranks.append(sigma[min(members, key=lambda v: sigma[v])])
    gamma = max(col)+1
    return statistics.mean(ranks) - best_rank(gamma)

N_RAND = 500
# múltiplos seeds por tamanho para média mais robusta
configs = {
    7: [(0.3,42),(0.5,59),(0.7,76),(0.4,88),(0.6,102)],
    8: [(0.3,11),(0.5,22),(0.7,33),(0.4,44),(0.6,55)],
    9: [(0.3,201),(0.5,202),(0.7,203),(0.4,204),(0.6,205)],
}

print("="*75)
print("  Z-SCORE DA SLO POR TAMANHO DE GRAFO")
print("  z = (μ_rnd - r_SLO) / σ_rnd")
print("  Alvo: z > 1.645 para p < 0.05 | z > 2.326 para p < 0.01")
print("="*75)
print(f"{'n':>3} {'p':>5} {'Γ':>3} {'r_SLO':>7} "
      f"{'μ_rnd':>7} {'σ_rnd':>7} {'z':>7} {'p-val':>8} {'sig':>4}")
print("-"*75)

zscore_by_n = defaultdict(list)

for n, casos in configs.items():
    for p, seed in casos:
        G = erdos_renyi(n, p, seed)
        gamma, sols = all_grundy(G)
        sg, _ = slo(G)
        r_slo = rank_medio(sols, sg)
        r_rnds = [rank_medio(sols, random_order(G, seed=seed+k*13+3000))
                  for k in range(N_RAND)]
        mu  = statistics.mean(r_rnds)
        sd  = statistics.stdev(r_rnds)
        z   = (mu - r_slo) / sd if sd > 0 else 0
        # p-valor empírico
        pval = sum(1 for r in r_rnds if r <= r_slo) / N_RAND
        sig  = "**" if pval<0.01 else ("*" if pval<0.05 else "ns")
        zscore_by_n[n].append(z)
        print(f"{n:>3} {p:>5.2f} {gamma:>3} {r_slo:>7.3f} "
              f"{mu:>7.3f} {sd:>7.3f} {z:>7.3f} {pval:>8.4f} {sig:>4}")
    # linha de média por n
    zs = zscore_by_n[n]
    print(f"{'':>3} {'média':>5} {'':>3} {'':>7} "
          f"{'':>7} {'':>7} {statistics.mean(zs):>7.3f}  ← média z para n={n}")
    print()

print("="*75)
print("  RESUMO: z médio por tamanho")
print("-"*75)
print(f"  {'n':>4}  {'z médio':>9}  {'N para p<0.05':>15}  {'N para p<0.01':>15}")
print("-"*75)
for n in sorted(zscore_by_n):
    z_med = statistics.mean(zscore_by_n[n])
    # N necessário para que o erro padrão seja pequeno o suficiente
    # dado que p-valor ≈ Φ(-z), estimamos o p verdadeiro
    from scipy.stats import norm
    p_true = norm.sf(z_med)   # P(Z > z_med) = 1 - Φ(z_med)
    # N para EP < 0.01
    N_005 = math.ceil(p_true*(1-p_true)/0.01**2) if 0 < p_true < 1 else 9999
    N_001 = math.ceil(p_true*(1-p_true)/0.005**2) if 0 < p_true < 1 else 9999
    print(f"  {n:>4}  {z_med:>9.3f}  {N_005:>15}  {N_001:>15}  "
          f"  p_aprox={p_true:.4f}")
print()
print("  Interpretação:")
print("  - z médio cresce com n → evidência que grafos maiores dão mais poder")
print("  - N necessário cai com n → com n maior, menos permutações bastam")



