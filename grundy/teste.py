import networkx as nx
from upper_bound import delta_1, delta_2, stair_factor, revised_stair_factor, psi_bound, psi_table
from vertex_ordering import smallest_last_ordering




if __name__ == "__main__":

    G = nx.erdos_renyi_graph(50, 0.5, 1)

    print( "Limite Delta1: ", delta_1(G) )
    print( "Limite Delta1: ", delta_2(G) )
    print( "stair factor: " , stair_factor(G) )
    print( "revised stair factor: " , revised_stair_factor(G) )
    print( "psi_bound: " , psi_bound(G, delta_1(G)) )

    
