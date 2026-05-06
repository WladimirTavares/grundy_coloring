"""
Análise estatística completa do teste de alinhamento semântico da SLO.

Para cada grafo:
1. Gera N=500 ordens aleatórias e calcula rank_medio para cada uma.
2. Testa normalidade da distribuição empírica (Shapiro-Wilk).
3. Se normal: usa teste-t de uma amostra (mais poderoso).
4. Se não normal: usa teste de Wilcoxon/permutação.
5. Plota histograma com marcação da SLO (ASCII).
"""

import random, itertools, statistics, math
from collections import defaultdict
from scipy import stats as sp

# ── utilitários ────────────────────────────────────────────────────────────

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
                degree[u] -= 1
                buckets[degree[u]].add(u)
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
    _, col = sols[0]
    gamma = max(col)+1
    return statistics.mean(ranks) - best_rank(gamma)

# ── histograma ASCII ────────────────────────────────────────────────────────

def ascii_hist(values, r_slo, n_bins=20, width=60):
    lo, hi = min(values), max(values)
    if hi == lo: hi = lo + 1
    bin_size = (hi - lo) / n_bins
    counts = [0]*n_bins
    for v in values:
        b = min(int((v - lo) / bin_size), n_bins-1)
        counts[b] += 1
    max_count = max(counts) or 1
    slo_bin = min(int((r_slo - lo) / bin_size), n_bins-1)

    print(f"    Distribuição de r̄ sob {len(values)} ordens aleatórias")
    print(f"    SLO (▼) = {r_slo:.3f}   μ = {statistics.mean(values):.3f}")
    print()
    for i, cnt in enumerate(counts):
        bar_len = int(cnt / max_count * width)
        bar = "█" * bar_len
        marker = " ◀ SLO" if i == slo_bin else ""
        lo_b = lo + i*bin_size
        print(f"  {lo_b:5.2f} │{bar}{marker}")
    print(f"       └{'─'*width}")
    print(f"         0{'contagem':>{width//2}}   {max_count}")
    print()

# ── análise principal ───────────────────────────────────────────────────────

def analisar(n, p, seed, N_RAND=500):
    G = erdos_renyi(n, p, seed)
    m = sum(len(v) for v in G.values())//2
    gamma, sols = all_grundy(G)
    sg, alpha = slo(G)
    r_slo = rank_medio(sols, sg)

    r_rnds = [
        rank_medio(sols, random_order(G, seed=seed+k*13+2000))
        for k in range(N_RAND)
    ]

    mu  = statistics.mean(r_rnds)
    sd  = statistics.stdev(r_rnds)

    # ── Teste de normalidade: Shapiro-Wilk ──────────────────────────────────
    # Shapiro-Wilk é o mais poderoso para n < 5000
    # H0: a distribuição é normal
    # p > 0.05 → não rejeitamos normalidade
    sw_stat, sw_p = sp.shapiro(r_rnds)

    # ── Teste de localização ────────────────────────────────────────────────
    # Se normal: t-test de uma amostra (testa se μ_rnd == r_slo)
    # Se não normal: teste de permutação empírico (mais robusto)
    normal = sw_p > 0.05

    if normal:
        # t-test: H0: μ_rnd = r_slo  vs  H1: μ_rnd > r_slo
        t_stat, t_p_two = sp.ttest_1samp(r_rnds, r_slo)
        # p unilateral (queremos μ_rnd > r_slo, i.e. t > 0)
        loc_p  = t_p_two / 2 if t_stat > 0 else 1 - t_p_two/2
        loc_method = "t-test"
        loc_stat   = f"t={t_stat:.3f}"
    else:
        # permutação empírica
        loc_p = sum(1 for r in r_rnds if r <= r_slo) / N_RAND
        loc_method = "permut."
        loc_stat   = "empírico"

    sig = "**" if loc_p < 0.01 else ("*" if loc_p < 0.05 else "ns")

    # ── assimetria e curtose ────────────────────────────────────────────────
    skew = sp.skew(r_rnds)
    kurt = sp.kurtosis(r_rnds)   # excesso de curtose (normal=0)

    # ── impressão ───────────────────────────────────────────────────────────
    sep = "═"*65
    print(sep)
    print(f"  Grafo: n={n}, p={p}, |E|={m}, α={alpha}, Γ={gamma}, #soluções={len(sols)}")
    print(sep)
    print(f"  r̄_SLO (normalizado)  = {r_slo:.4f}")
    print(f"  μ_rnd                = {mu:.4f}")
    print(f"  σ_rnd                = {sd:.4f}")
    print(f"  Δ = μ_rnd - r̄_SLO   = {mu-r_slo:+.4f}")
    print()
    print(f"  ── Normalidade (Shapiro-Wilk, N={N_RAND}) ──")
    print(f"     W={sw_stat:.4f},  p={sw_p:.4f}")
    print(f"     Assimetria (skew) = {skew:.3f}  "
          f"{'≈ simétrica' if abs(skew)<0.5 else 'assimétrica'}")
    print(f"     Curtose (excesso) = {kurt:.3f}  "
          f"{'≈ normal' if abs(kurt)<1 else 'caudas pesadas' if kurt>0 else 'caudas leves'}")
    print(f"     Distribuição: {'NORMAL ✓' if normal else 'NÃO NORMAL ✗'}")
    print()
    print(f"  ── Teste de localização ({loc_method}) ──")
    print(f"     H0: SLO não é melhor que uma ordem aleatória")
    print(f"     H1: SLO produz rank médio menor (unilateral)")
    print(f"     {loc_stat},  p={loc_p:.4f}  {sig}")
    print(f"     {'Rejeitamos H0' if loc_p < 0.05 else 'Não rejeitamos H0'} "
          f"ao nível 5%")
    print()
    ascii_hist(r_rnds, r_slo)


if __name__ == "__main__":
    print("\n" + "█"*65)
    print("  ANÁLISE ESTATÍSTICA DO ALINHAMENTO SEMÂNTICO DA SLO")
    print("█"*65 + "\n")

    size = 10
    configs = [
        (size, 0.10, 42),
        (size, 0.25, 59),
        (size, 0.50, 76),
        (size, 0.75, 93),
        (size, 0.90, 110),
    ]
    for n, p, seed in configs:
        analisar(n, p, seed, N_RAND=500)


"""
████████████████████████████████████████████████████████████████
  ANÁLISE ESTATÍSTICA DO ALINHAMENTO SEMÂNTICO DA SLO
█████████████████████████████████████████████████████████████████

═════════════════════════════════════════════════════════════════
  Grafo: n=9, p=0.1, |E|=7, α=1, Γ=3, #soluções=181440
═════════════════════════════════════════════════════════════════
  r̄_SLO (normalizado)  = 0.5611
  μ_rnd                = 0.9441
  σ_rnd                = 0.2702
  Δ = μ_rnd - r̄_SLO   = +0.3830

  ── Normalidade (Shapiro-Wilk, N=500) ──
     W=0.9954,  p=0.1500
     Assimetria (skew) = 0.008  ≈ simétrica
     Curtose (excesso) = -0.432  ≈ normal
     Distribuição: NORMAL ✓

  ── Teste de localização (t-test) ──
     H0: SLO não é melhor que uma ordem aleatória
     H1: SLO produz rank médio menor (unilateral)
     t=31.693,  p=0.0000  **
     Rejeitamos H0 ao nível 5%

    Distribuição de r̄ sob 500 ordens aleatórias
    SLO (▼) = 0.561   μ = 0.944

   0.17 │██
   0.24 │
   0.32 │█████
   0.40 │████████████
   0.47 │█████████████████
   0.55 │███████████████████████ ◀ SLO
   0.62 │████████████████████████████████████████
   0.70 │███████████████████████████████████████████████
   0.78 │█████████████████████████████████████████████████████
   0.85 │████████████████████████████████████████████████████████████
   0.93 │███████████████████████████████████████████████████
   1.00 │████████████████████████████████████████████████████
   1.08 │███████████████████████████████████████████████
   1.16 │█████████████████████████████████████
   1.23 │███████████████████████████
   1.31 │███████████████████████████
   1.38 │███████████████████
   1.46 │█████
   1.54 │███
   1.61 │█
       └────────────────────────────────────────────────────────────
         0                      contagem   56

═════════════════════════════════════════════════════════════════
  Grafo: n=9, p=0.25, |E|=8, α=1, Γ=4, #soluções=4158
═════════════════════════════════════════════════════════════════
  r̄_SLO (normalizado)  = 0.0000
  μ_rnd                = 1.1655
  σ_rnd                = 0.6998
  Δ = μ_rnd - r̄_SLO   = +1.1655

  ── Normalidade (Shapiro-Wilk, N=500) ──
     W=0.9700,  p=0.0000
     Assimetria (skew) = 0.134  ≈ simétrica
     Curtose (excesso) = -0.641  ≈ normal
     Distribuição: NÃO NORMAL ✗

  ── Teste de localização (permut.) ──
     H0: SLO não é melhor que uma ordem aleatória
     H1: SLO produz rank médio menor (unilateral)
     empírico,  p=0.0940  ns
     Não rejeitamos H0 ao nível 5%

    Distribuição de r̄ sob 500 ordens aleatórias
    SLO (▼) = 0.000   μ = 1.165

   0.00 │██████████████████████████████████████ ◀ SLO
   0.15 │█████████████████████
   0.30 │
   0.45 │█████████████████████████████████████████
   0.60 │
   0.75 │█████████████████████████████████████
   0.90 │████████████████████████████████████████████████████████████
   1.05 │
   1.20 │████████████████████████████████████████████████
   1.35 │
   1.50 │████████████████████████████████████████████████████████
   1.65 │█████████████████████████████████
   1.80 │
   1.95 │███████████████████████████
   2.10 │
   2.25 │████████████████████████
   2.40 │██████████
   2.55 │
   2.70 │████
   2.85 │█
       └────────────────────────────────────────────────────────────
         0                      contagem   74

═════════════════════════════════════════════════════════════════
  Grafo: n=9, p=0.5, |E|=19, α=3, Γ=6, #soluções=325
═════════════════════════════════════════════════════════════════
  r̄_SLO (normalizado)  = 0.0000
  μ_rnd                = 0.6703
  σ_rnd                = 0.4079
  Δ = μ_rnd - r̄_SLO   = +0.6703

  ── Normalidade (Shapiro-Wilk, N=500) ──
     W=0.9678,  p=0.0000
     Assimetria (skew) = 0.202  ≈ simétrica
     Curtose (excesso) = -0.526  ≈ normal
     Distribuição: NÃO NORMAL ✗

  ── Teste de localização (permut.) ──
     H0: SLO não é melhor que uma ordem aleatória
     H1: SLO produz rank médio menor (unilateral)
     empírico,  p=0.0920  ns
     Não rejeitamos H0 ao nível 5%

    Distribuição de r̄ sob 500 ordens aleatórias
    SLO (▼) = 0.000   μ = 0.670

   0.00 │████████████████████████████████ ◀ SLO
   0.09 │████████████████████████████
   0.18 │
   0.27 │██████████████████████████████████████████
   0.37 │
   0.46 │███████████████████████████████████████████████
   0.55 │
   0.64 │████████████████████████████████████████████████████████████
   0.73 │
   0.82 │████████████████████████████████████████
   0.92 │█████████████████████████████████████████████████
   1.01 │
   1.10 │████████████████████████████
   1.19 │
   1.28 │██████████████
   1.37 │
   1.47 │████████
   1.56 │
   1.65 │███
   1.74 │█
       └────────────────────────────────────────────────────────────
         0                      contagem   84

═════════════════════════════════════════════════════════════════
  Grafo: n=9, p=0.75, |E|=26, α=5, Γ=7, #soluções=4320
═════════════════════════════════════════════════════════════════
  r̄_SLO (normalizado)  = 0.0000
  μ_rnd                = 0.5131
  σ_rnd                = 0.3531
  Δ = μ_rnd - r̄_SLO   = +0.5131

  ── Normalidade (Shapiro-Wilk, N=500) ──
     W=0.9547,  p=0.0000
     Assimetria (skew) = 0.416  ≈ simétrica
     Curtose (excesso) = -0.425  ≈ normal
     Distribuição: NÃO NORMAL ✗

  ── Teste de localização (permut.) ──
     H0: SLO não é melhor que uma ordem aleatória
     H1: SLO produz rank médio menor (unilateral)
     empírico,  p=0.1260  ns
     Não rejeitamos H0 ao nível 5%

    Distribuição de r̄ sob 500 ordens aleatórias
    SLO (▼) = 0.000   μ = 0.513

   0.00 │█████████████████████████████████████████████ ◀ SLO
   0.08 │███████████████████████████████
   0.16 │
   0.24 │████████████████████████████████████████████████████████████
   0.31 │
   0.39 │███████████████████████████████████████████████
   0.47 │
   0.55 │██████████████████████████████████████████████
   0.63 │
   0.71 │███████████████████████████████████████████████
   0.79 │█████████████████████████████████
   0.86 │
   0.94 │██████████████████
   1.02 │
   1.10 │█████████████████
   1.18 │
   1.26 │█████
   1.34 │
   1.41 │███
   1.49 │█
       └────────────────────────────────────────────────────────────
         0                      contagem   84


"""