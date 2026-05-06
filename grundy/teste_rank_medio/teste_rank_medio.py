import random, itertools, statistics
from collections import defaultdict
from math import comb

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
    removed = [False]*n
    removal_order = []
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
                buckets[degree[u]].discard(u); degree[u] -= 1; buckets[degree[u]].add(u)
    sigma = [0]*n
    for rank, v in enumerate(reversed(removal_order)): sigma[v] = rank
    return sigma, alpha

def random_order(G, seed):
    n = len(G); perm = list(range(n)); random.Random(seed).shuffle(perm)
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

def best_rank(gamma: int):
    ranks  = []
    for i in range(gamma):
        ranks.append(i)
    return statistics.mean(ranks)
    

def rank_medio(sols, sigma):
    ranks = []
    for _,col in sols:
        cls = defaultdict(list)
        for v,c in enumerate(col): cls[c].append(v)
        for members in cls.values():
            ranks.append(sigma[min(members, key=lambda v: sigma[v])])
    _, col = sols[0]
    gamma = max(col)+1
    return statistics.mean(ranks) - best_rank(gamma)

# Rodar apenas n=7 e n=8, 200 aleatórias
size = 10
configs = [(size,0.1,42),(size,0.25,59),(size,0.5,76),(size,0.75,93),(size,0.9,110)]
N_RAND = 500

print("{:>3} {:>4} {:>3} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>7} {:>4} {:>7}".format(
    'n','p','Γ','#sol','r_SLO','μ_rnd','σ_rnd','min','max','p-val','sig','Δ'))
print('-'*80)
for n,p,seed in configs:
    G = erdos_renyi(n,p,seed)
    gamma, sols = all_grundy(G)
    sg, _ = slo(G)
    r_slo = rank_medio(sols, sg)
    r_rnds = [rank_medio(sols, random_order(G, seed=seed+k*13+2000)) for k in range(N_RAND)]
    mu = statistics.mean(r_rnds); sd = statistics.stdev(r_rnds)
    mn = min(r_rnds); mx = max(r_rnds)
    pval = sum(1 for r in r_rnds if r <= r_slo)/N_RAND
    sig = '**' if pval<0.01 else ('*' if pval<0.05 else 'ns')
    delta = mu - r_slo
    print(f'{n:>3} {p:>4.2f} {gamma:>3} {len(sols):>7} {r_slo:>7.3f} {mu:>7.3f} {sd:>6.3f} {mn:>6.3f} {mx:>6.3f} {pval:>7.3f} {sig:>4} {delta:>+7.3f}')
print()
print('Legenda: ** p<0.01  * p<0.05  ns=nao significativo  Delta=mu_rnd - r_SLO (>0: SLO melhor)')

"""
RAND = 200
  n    p   Γ    #sol   r_SLO   μ_rnd  σ_rnd    min    max   p-val  sig       Δ
--------------------------------------------------------------------------------
  9 0.10   3  181440   0.561   0.946  0.280  0.167  1.689   0.095   ns  +0.384
  9 0.25   4    4158   0.000   1.206  0.698  0.000  2.750   0.075   ns  +1.206
  9 0.50   6     325   0.000   0.692  0.412  0.000  1.667   0.090   ns  +0.692
  9 0.75   7    4320   0.000   0.484  0.358  0.000  1.571   0.155   ns  +0.484
  9 0.90   7   80640   0.000   0.526  0.294  0.000  1.286   0.060   ns  +0.526

Legenda: ** p<0.01  * p<0.05  ns=nao significativo  Delta=mu_rnd - r_SLO (>0: SLO melhor)


N_RAND = 500

  n    p   Γ    #sol   r_SLO   μ_rnd  σ_rnd    min    max   p-val  sig       Δ
--------------------------------------------------------------------------------
  9 0.10   3  181440   0.561   0.944  0.270  0.167  1.689   0.082   ns  +0.383
  9 0.25   4    4158   0.000   1.165  0.700  0.000  3.000   0.094   ns  +1.165
  9 0.50   6     325   0.000   0.670  0.408  0.000  1.833   0.092   ns  +0.670
  9 0.75   7    4320   0.000   0.513  0.353  0.000  1.571   0.126   ns  +0.513
  9 0.90   7   80640   0.000   0.528  0.297  0.000  1.286   0.060   ns  +0.528

Legenda: ** p<0.01  * p<0.05  ns=nao significativo  Delta=mu_rnd - r_SLO (>0: SLO melhor)


"""

