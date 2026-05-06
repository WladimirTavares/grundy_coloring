import functools
import time
import tracemalloc
import networkx as nx
from greedy_coloring import is_greedy_coloring
from bb import branch_and_bound
from formulation import solver_carvalho_representante2
from vertex_ordering import smallest_last_ordering
from upper_bound import upper_bound1
from lower_bound import lb_reverse_lf, lb_reverse_sl
import math

def create_bit_graph(G : nx.Graph):
    nodes = sorted(G.nodes())
    idx   = {v: i for i, v in enumerate(nodes)}
    n     = len(nodes)

    # adj[i]: inteiro cujo bit j está ativo sse existe aresta i–j
    adj = [0] * n
    for u in G:
        for v in G[u]:
            adj[idx[u]] |= (1 << idx[v])

    FULL = (1 << n) - 1  # bitmask com todos os n vértices ativos

    # non_adj[i]: bitmask dos NÃO-vizinhos de i (excluindo o próprio i)
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = (~adj[v]) & FULL & ~(1 << v)


    return adj, non_adj

def expand(non_adj: list, R: int, P: int, X: int):
    
        
    if P == 0 and X == 0:
        yield R
        return

    u_mask = P | X
    if u_mask == 0:
        return

    u = (u_mask & -u_mask).bit_length() - 1

    candidates = P & ~non_adj[u]

    
    while candidates:
        v = (candidates & -candidates).bit_length() - 1
        candidates &= candidates - 1

        yield from expand(
            non_adj,
            R | (1 << v),
            P & non_adj[v],
            X & non_adj[v],
        )

        P &= ~(1 << v)
        X |= (1 << v)

def delta_1_bitmask(adj : list, S: int) -> int:
        """Grau máximo Δ(G[S]) do subgrafo induzido por S."""
        max_deg = 0
        T = S
        while T:
            v      = (T & -T).bit_length() - 1   # bit menos significativo
            T     &= T - 1                        # remove v de T
            deg    = bin(adj[v] & S).count('1')   # grau de v em G[S]
            if deg > max_deg:
                max_deg = deg
        return max_deg + 1





def bb_bitmask1(
        G: nx.Graph,
        time_limit = math.inf
):
    nodes = sorted(G.nodes())
    n     = len(nodes)

    adj, non_adj = create_bit_graph(G)
    
    C = []
    FULL = (1 << n) - 1  # bitmask com todos os n vértices ativos
    bb_nodes = 0
    pruned = 0

    lb1 = lb_reverse_sl(G)
    lb2 = lb_reverse_lf(G)
    LB    = max(lb1["lower_bound"], lb2["lower_bound"])
    bestC = lb1["coloring"] if lb1["lower_bound"] >= lb2["lower_bound"] else lb2["coloring"]
    
    start = time.time()
    deadline = start + time_limit


    def solve(S: int) -> int:
        nonlocal bb_nodes, LB, bestC, pruned, deadline
        
        if time.time () > deadline:
            return 0


        
        if S == 0:
            if len(C) > LB:
                LB    = len(C)
                bestC = C.copy()
            return len(C)
        else:
            
            sizeS = bin(S).count('1')
            
            if len(C) + sizeS <= LB:
                pruned += 1
                return 0

            if len(C) + delta_1_bitmask(adj, S) <= LB:
                pruned += 1
                return 0



            bb_nodes += 1


            max_val = 0
            for R in expand(non_adj, 0, S, 0):
                C.append(R)
                S_next = S & ~R
                val = 1 + solve(S_next)
                max_val = max(max_val, val)    
                C.pop()
            return max_val

    
    solve(FULL)

    #tracemalloc.start()
    #start = time.time()

    best  = solve(FULL)
    end   = time.time()
    #current, peak = tracemalloc.get_traced_memory()
    #tracemalloc.stop()

    #info  = solve.cache_info()
    #total = info.hits + info.misses

    coloring = []
    for R in bestC:
        color = [ nodes[i] for i in range(n) if R & (1 << i)]
        coloring.append(color)

    return {
        "model":        "BITBB",
        "gamma":        LB,
        "cpu_s":        end - start,
        "classes":      coloring,
        "valid":        is_greedy_coloring(G, coloring),
        "bb_nodes":     bb_nodes,
        "pruned":       pruned,
        #"hits":         info.hits,
        #"misses":       info.misses,
        #"hit_rate":     info.hits / total if total > 0 else 0,
        #"memoria_pico": peak,
    }
    
def pick_pivot(P, X, non_adj):
    u_mask = P | X
    best = -1
    best_deg = -1
    tmp = u_mask
    while tmp:
        u = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        deg = (P & non_adj[u]).bit_count()
        if deg > best_deg:
            best_deg = deg
            best = u
    return best

def expand2(non_adj: list, R: int, P: int, X: int):
    
        
    if P == 0 and X == 0:
        yield R
        return

    u_mask = P | X
    if u_mask == 0:
        return

    u = pick_pivot(P, X, non_adj)
    candidates = P & ~non_adj[u]

    
    while candidates:
        v = (candidates & -candidates).bit_length() - 1
        candidates &= candidates - 1

        yield from expand2(
            non_adj,
            R | (1 << v),
            P & non_adj[v],
            X & non_adj[v],
        )

        P &= ~(1 << v)
        X |= (1 << v)



def bb_bitmask2(
        G: nx.Graph,
        time_limit = math.inf
):

    nodes = sorted(G.nodes())
    n     = len(nodes)

    adj, non_adj = create_bit_graph(G)
    
    C = []
    FULL = (1 << n) - 1  # bitmask com todos os n vértices ativos
    bb_nodes = 0
    pruned = 0

    lb1 = lb_reverse_sl(G)
    lb2 = lb_reverse_lf(G)
    LB    = max(lb1["lower_bound"], lb2["lower_bound"])
    bestC = lb1["coloring"] if lb1["lower_bound"] >= lb2["lower_bound"] else lb2["coloring"]
    

    start = time.time()
    deadline = start + time_limit


    def solve(S: int) -> int:
        nonlocal bb_nodes, LB, bestC, pruned, deadline

        if time.time () > deadline:
            return 0
        
        
        if S == 0:
            if len(C) > LB:
                #print(C)
                LB    = len(C)
                bestC = C.copy()
            return len(C)
        else:
            
            sizeS = bin(S).count('1')
            
            if len(C) + sizeS <= LB:
                pruned += 1
                return 0

            if len(C) + delta_1_bitmask(adj, S)  <= LB:
                pruned += 1
                return 0



            bb_nodes += 1


            max_val = 0
            for R in expand2(non_adj, 0, S, 0):
                C.append(R)
                S_next = S & ~R
                val = 1 + solve(S_next)
                max_val = max(max_val, val)    
                C.pop()
            return max_val

    
    solve(FULL)

    #tracemalloc.start()
    #start = time.time()
    
    best  = solve(FULL)
    end   = time.time()
    #current, peak = tracemalloc.get_traced_memory()
    #tracemalloc.stop()

    #info  = solve.cache_info()
    #total = info.hits + info.misses

    coloring = []
    for R in bestC:
        color = [ nodes[i] for i in range(n) if R & (1 << i)]
        coloring.append(color)

    return {
        "model":        "BITBB",
        "gamma":        LB,
        "cpu_s":        end - start,
        "classes":      coloring,
        "valid":        is_greedy_coloring(G, coloring),
        "bb_nodes":     bb_nodes,
        "pruned":       pruned,
        #"hits":         info.hits,
        #"misses":       info.misses,
        #"hit_rate":     info.hits / total if total > 0 else 0,
        #"memoria_pico": peak,
    }


def stair_factor_mask(adj, S):
    """
    Upper bound do Grundy para subgrafo induzido por S (bitmask)
    adj: lista de bitmasks
    S: bitmask de vértices ativos
    """
    if S == 0:
        return 0

    # lista de vértices em S
    vertices = []
    tmp = S
    while tmp:
        v = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        vertices.append(v)

    n = len(vertices)

    # graus locais
    deg = {}
    for v in vertices:
        deg[v] = (adj[v] & S).bit_count()

    # buckets
    buckets = [0] * (n + 1)
    for v in vertices:
        buckets[deg[v]] |= (1 << v)

    max_degree = max(deg.values()) if deg else 0

    alive = S
    residue = []

    for _ in range(n):
        while max_degree >= 0 and buckets[max_degree] == 0:
            max_degree -= 1

        d = max_degree

        u = (buckets[d] & -buckets[d]).bit_length() - 1
        buckets[d] &= ~(1 << u)

        residue.append(d)

        neighbors = adj[u] & alive
        while neighbors:
            v = (neighbors & -neighbors).bit_length() - 1
            neighbors &= neighbors - 1

            dv = deg[v]
            buckets[dv] &= ~(1 << v)
            deg[v] -= 1
            buckets[dv - 1] |= (1 << v)

        alive &= ~(1 << u)
        del deg[u]

    k = float('inf')
    for i, d in enumerate(residue):
        k = min(k, d + i + 1)

    return int(k)

def delta_2_mask(adj: list[int], S: int) -> int:
    """
    Versão bitmask do bound δ₂ para o subgrafo induzido por S.

    Para cada vértice v ∈ S, considera vizinhos u ∈ N(v) ∩ S tais que:
        deg(u) ≤ deg(v)

    e retorna:
        max_v max_{u} deg(u) + 1

    Parâmetros
    ----------
    adj : list[int]
        Lista de bitmasks de adjacência.
    S : int
        Bitmask representando o subconjunto de vértices.

    Retorno
    -------
    int
        Valor do bound δ₂(G[S]).

    Complexidade
    ------------
    O(m_S), onde m_S é o número de arestas no subgrafo induzido.
    """

    deg = {}
    T = S
    while T:
        v = (T & -T).bit_length() - 1
        T &= T - 1
        deg[v] = (adj[v] & S).bit_count()

    max_val = 0

    T = S
    while T:
        v = (T & -T).bit_length() - 1
        T &= T - 1

        dv = deg[v]
        best_local = 0

        neighbors = adj[v] & S
        while neighbors:
            u = (neighbors & -neighbors).bit_length() - 1
            neighbors &= neighbors - 1

            du = deg[u]
            if du <= dv and du > best_local:
                best_local = du

        if best_local > max_val:
            max_val = best_local

    return max_val + 1


def bb_bitmask3(
    G: nx.Graph,
    time_limit : int = math.inf
):
    nodes = sorted(G.nodes())
    n     = len(nodes)

    adj, non_adj = create_bit_graph(G)
    
    C = []
    FULL = (1 << n) - 1  # bitmask com todos os n vértices ativos
    bb_nodes = 0
    pruned = 0

    lb1 = lb_reverse_sl(G)
    lb2 = lb_reverse_lf(G)
    LB    = max(lb1["lower_bound"], lb2["lower_bound"])
    bestC = lb1["coloring"] if lb1["lower_bound"] >= lb2["lower_bound"] else lb2["coloring"]
    
    start = time.time()
    deadline = start + time_limit



    def solve(S: int) -> int:
        nonlocal bb_nodes, LB, bestC, pruned

        if time.time () > deadline:
            return 0
        
        
        if S == 0:
            if len(C) > LB:
                #print(C)
                LB    = len(C)
                bestC = C.copy()
            return len(C)
        else:
            
            sizeS = bin(S).count('1')
            
            if len(C) + sizeS <= LB:
                pruned += 1
                return 0

            if len(C) + delta_1_bitmask(adj, S) <= LB:
                pruned += 1
                return 0

            if len(C) + stair_factor_mask(adj, S) <= LB:
                pruned += 1
                return 0


            bb_nodes += 1


            max_val = 0
            for R in expand2(non_adj, 0, S, 0):
                C.append(R)
                S_next = S & ~R
                val = 1 + solve(S_next)
                max_val = max(max_val, val)    
                C.pop()
            return max_val

    
    solve(FULL)

    #tracemalloc.start()
    #start = time.time()
    best  = solve(FULL)
    end   = time.time()
    #current, peak = tracemalloc.get_traced_memory()
    #tracemalloc.stop()

    #info  = solve.cache_info()
    #total = info.hits + info.misses

    coloring = []
    for R in bestC:
        color = [ nodes[i] for i in range(n) if R & (1 << i)]
        coloring.append(color)

    return {
        "model":        "BITBB",
        "gamma":        LB,
        "cpu_s":        end - start,
        "classes":      coloring,
        "valid":        is_greedy_coloring(G, coloring),
        "bb_nodes":     bb_nodes,
        "pruned":       pruned,
        #"hits":         info.hits,
        #"misses":       info.misses,
        #"hit_rate":     info.hits / total if total > 0 else 0,
        #"memoria_pico": peak,
    }


def bb_bitmask4(
    G: nx.Graph,
    time_limit : int = math.inf   
):
    nodes = sorted(G.nodes())
    n     = len(nodes)

    adj, non_adj = create_bit_graph(G)
    
    C = []
    FULL = (1 << n) - 1  # bitmask com todos os n vértices ativos
    bb_nodes = 0
    pruned = 0

    lb1 = lb_reverse_sl(G)
    lb2 = lb_reverse_lf(G)
    LB    = max(lb1["lower_bound"], lb2["lower_bound"])
    bestC = lb1["coloring"] if lb1["lower_bound"] >= lb2["lower_bound"] else lb2["coloring"]
    
    start = time.time()
    deadline = start + time_limit

    change = False

    def solve(S: int) -> int:
        nonlocal bb_nodes, LB, bestC, pruned, deadline, change

        if time.time () > deadline:
            return 0
    
        
        if S == 0:
            if len(C) > LB:
                #print(C)
                change = True
                LB    = len(C)
                bestC = C.copy()
            return len(C)
        else:
            
            sizeS = bin(S).count('1')
            
            if len(C) + sizeS <= LB:
                pruned += 1
                return 0

            if len(C) + delta_1_bitmask(adj, S) <= LB:
                pruned += 1
                return 0

            if len(C) + delta_2_mask(adj, S) <= LB:
                pruned += 1
                return 0

            if len(C) + stair_factor_mask(adj, S) <= LB:
                pruned += 1
                return 0


            bb_nodes += 1


            max_val = 0
            for R in expand2(non_adj, 0, S, 0):
                C.append(R)
                S_next = S & ~R
                val = 1 + solve(S_next)
                max_val = max(max_val, val)    
                C.pop()
            return max_val

    
    solve(FULL)

    #tracemalloc.start()
    #start = time.time()

    best  = solve(FULL)
    end   = time.time()
    #current, peak = tracemalloc.get_traced_memory()
    #tracemalloc.stop()

    #info  = solve.cache_info()
    #total = info.hits + info.misses

    if change:
        coloring = []
        for R in bestC:
            color = [ nodes[i] for i in range(n) if R & (1 << i)]
            coloring.append(color)
    else:
        coloring = bestC
        
    return {
        "model":        "BITBB",
        "gamma":        LB,
        "cpu_s":        end - start,
        "classes":      coloring,
        "valid":        is_greedy_coloring(G, coloring),
        "bb_nodes":     bb_nodes,
        "pruned":       pruned,
        #"hits":         info.hits,
        #"misses":       info.misses,
        #"hit_rate":     info.hits / total if total > 0 else 0,
        #"memoria_pico": peak,
    }


def dp_grundy(
    G: nx.Graph,
    maxsize : int = None,
    time_limit : int = math.inf   
):
    
    nodes = sorted(G.nodes())
    n     = len(nodes)

    adj, non_adj = create_bit_graph(G)
    
    C = []
    FULL = (1 << n) - 1  # bitmask com todos os n vértices ativos
    bb_nodes = 0
    pruned = 0

    choice = {}
    
    @functools.lru_cache(maxsize=maxsize)
    def solve(S: int) -> int:
        nonlocal pruned, bb_nodes
        if S == 0:
            return 0
        else:
            bb_nodes += 1
            max_val = 0
            for R in expand2(non_adj, 0, S, 0):
                C.append(R)
                S_next = S & ~R

                val = 1 + solve(S_next)
                if val > max_val:
                    max_val = val
                    choice[S] = R
                C.pop()
            return max_val

    

    tracemalloc.start()
    start = time.time()

    LB = solve(FULL)
    end   = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    info  = solve.cache_info()
    total = info.hits + info.misses

    S = FULL
    coloring = []
        
    while S != 0:
        R = choice[S]
        color = [ nodes[i] for i in range(n) if R & (1 << i)]
        coloring.append(color)
        S = S & ~R
        

    return {
        "model":        "DPBIT",
        "gamma":        LB,
        "cpu_s":        end - start,
        "classes":      coloring,
        "valid":        is_greedy_coloring(G, coloring),
        "bb_nodes":     bb_nodes,
        "pruned":       pruned,
        "hits":         info.hits,
        "misses":       info.misses,
        "hit_rate":     info.hits / total if total > 0 else 0,
        "memoria_pico": peak,
    }


def dp_grundy2(
    G: nx.Graph,
    maxsize : int = None,
    time_limit : int = math.inf   
):
    
    nodes = sorted(G.nodes())
    n     = len(nodes)

    adj, non_adj = create_bit_graph(G)
    
    C = []
    FULL = (1 << n) - 1  # bitmask com todos os n vértices ativos
    bb_nodes = 0
    pruned = 0

    choice = {}
    table  = {}
    
    def solve(S: int) -> int:
        nonlocal pruned, bb_nodes
        if S == 0:
            return 0
        elif S in table:
            return table[S] 
        else:
            bb_nodes += 1
            max_val = 0
            for R in expand2(non_adj, 0, S, 0):
                C.append(R)
                S_next = S & ~R

                val = 1 + solve(S_next)
                if val > max_val:
                    max_val = val
                    choice[S] = R
                C.pop()
            table[S] = max_val
            return max_val

    

    tracemalloc.start()
    start = time.time()

    LB = solve(FULL)
    end   = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()


    S = FULL
    coloring = []
        
    while S != 0:
        R = choice[S]
        color = [ nodes[i] for i in range(n) if R & (1 << i)]
        coloring.append(color)
        S = S & ~R
        

    return {
        "model":        "DPBIT",
        "gamma":        LB,
        "cpu_s":        end - start,
        "classes":      coloring,
        "valid":        is_greedy_coloring(G, coloring),
        "bb_nodes":     bb_nodes,
        "pruned":       pruned,
    }


from upper_bound import stair_factor, delta_2
from bb import branch_and_bound

if __name__ == "__main__":

    G = nx.erdos_renyi_graph(20, 0.5, 42)

    #adj, non_adj = create_bit_graph(G)

    
    #for S in expand(0, )

    r = bb_bitmask1(G)
    print(r)

    r = bb_bitmask2(G)
    print(r)

    r = bb_bitmask3(G)
    print(r)

    r = bb_bitmask4(G)
    print(r)



    # for i in range(10):
    #     G = nx.erdos_renyi_graph(25, 0.5, i)
    #     r1 = bb_bitmask4(G) # versao com bitmap e todos os cortes
    #     r2 = dp_grundy(G) 
    #     r3 = dp_grundy2(G) 

    #     print(r1)
    #     print(r2)
    #     print(r3)
    #     print()

        
    #     if r1["gamma"] != r2["gamma"]:
    #         print(r1)
    #         print(r2)
        
    #         print("dp grundy 1 errada")
    #         break 
    #     if r1["gamma"] != r3["gamma"]:
    #         print(r1)
    #         print(r3)
        
    #         print("dp grundy 2 errada")
    #         break
        

    # print( branch_and_bound(G)) # versao sem bitmap
    # print( bb_bitmask1(G))
    # print( bb_bitmask2(G))
    # print( bb_bitmask3(G))
    
    
    # print( branch_and_bound(G, 20)) # versao sem bitmap com tempo limite 20s
    # print( bb_bitmask1(G, 20))
    # print( bb_bitmask2(G, 20))
    # print( bb_bitmask3(G, 20))
    # print( bb_bitmask4(G, 20)) # versao com bitmap e todos os cortes com tempo limite
    

