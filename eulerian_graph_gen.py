import itertools
import networkx as nx
import random

def create_complex_eulerian_digraph(num_nodes=50, num_edges=250, max_cycles=1000):
    """
    Generates a complex Eulerian directed graph by creating strongly connected 
    cycle components and enforcing that in-degree == out-degree for all nodes.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if num_edges < num_nodes:
        raise ValueError("Number of edges must be at least the number of nodes.")
    max_edges = num_nodes * num_nodes
    if num_edges > max_edges:
        raise ValueError(
            f"For a directed graph with {num_nodes} nodes (including self-loops), the maximum number of edges is {max_edges}."
        )

    G = nx.DiGraph()
    
    # 1. Guarantee a base strongly-connected component (Hamiltonian Cycle)
    # This ensures the graph is weakly and strongly connected.
    nodes = list(range(num_nodes))
    for i in range(len(nodes)):
        u = nodes[i]
        v = nodes[(i + 1) % len(nodes)]
        G.add_edge(u, v)

    remaining_edges = num_edges - G.number_of_edges()

    def add_self_loop():
        for node in nodes:
            if not G.has_edge(node, node):
                G.add_edge(node, node)
                return True
        return False

    def add_3_cycle():
        for _ in range(max_cycles):
            cycle_nodes = random.sample(nodes, 3)
            cycle_edges = [
                (cycle_nodes[i], cycle_nodes[(i + 1) % 3])
                for i in range(3)
            ]
            if all(not G.has_edge(u, v) for u, v in cycle_edges):
                for u, v in cycle_edges:
                    G.add_edge(u, v)
                return True
        for u, v, w in itertools.permutations(nodes, 3):
            if not G.has_edge(u, v) and not G.has_edge(v, w) and not G.has_edge(w, u):
                G.add_edge(u, v)
                G.add_edge(v, w)
                G.add_edge(w, u)
                return True
        return False

    def add_2_cycle():
        for _ in range(max_cycles):
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                G.add_edge(u, v)
                G.add_edge(v, u)
                return True
        for u, v in itertools.permutations(nodes, 2):
            if u != v and not G.has_edge(u, v) and not G.has_edge(v, u):
                G.add_edge(u, v)
                G.add_edge(v, u)
                return True
        return False

    if remaining_edges == 1:
        if not add_self_loop():
            raise RuntimeError("Unable to add a self-loop to satisfy the final edge.")
        remaining_edges -= 1

    if remaining_edges % 2 == 1:
        if remaining_edges < 3:
            raise RuntimeError("Cannot satisfy an odd remaining edge count without at least 3 edges.")
        if not add_3_cycle():
            raise RuntimeError("Unable to add a new 3-cycle to satisfy the odd remaining edge count.")
        remaining_edges -= 3

    while remaining_edges > 0:
        if not add_2_cycle():
            raise RuntimeError("Unable to add a new 2-cycle to reach the desired edge count.")
        remaining_edges -= 2
    
    # Verification
    assert nx.is_eulerian(G), "Graph is not Eulerian!"
    
    return G


# # --- Example Usage ---

G = create_complex_eulerian_digraph(num_nodes=30, num_edges=150)
print(f"Generated a complex Eulerian DiGraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print(f"Is Eulerian: {nx.is_eulerian(G)}")