import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def grover_coin(d):
    """
    Return the d x d Grover diffusion coin.
    """
    if d == 1:
        return np.array([[1.0]], dtype=complex)

    J = np.ones((d, d), dtype=complex)
    I = np.eye(d, dtype=complex)
    return (2 / d) * J - I


def check_eulerian(edges):
    """
    Check whether a directed graph is Eulerian.
    edges is a list of tuples: [(u,v), ...]
    """
    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    vertices = set()

    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1
        vertices.add(u)
        vertices.add(v)

    for v in vertices:
        if indeg[v] != outdeg[v]:
            return False

    return True


def build_eulerian_walk_operator(edges):
    """
    Build the edge-based Eulerian quantum walk operator U.

    Basis states are directed edges.
    Each incoming edge to a vertex scatters into outgoing edges from that vertex.
    """
    if not check_eulerian(edges):
        raise ValueError("Graph is not Eulerian.")

    edge_to_index = {e: i for i, e in enumerate(edges)}
    n_edges = len(edges)

    incoming = defaultdict(list)
    outgoing = defaultdict(list)

    for e in edges:
        u, v = e
        outgoing[u].append(e)
        incoming[v].append(e)

    U = np.zeros((n_edges, n_edges), dtype=complex)

    vertices = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)

    for v in vertices:
        in_edges = incoming[v]
        out_edges = outgoing[v]

        d = len(in_edges)
        C = grover_coin(d)

        for i, e_in in enumerate(in_edges):
            for j, e_out in enumerate(out_edges):
                row = edge_to_index[e_out]
                col = edge_to_index[e_in]
                U[row, col] = C[j, i]

    return U, edge_to_index

edges = [
    (0, 1),
    (1, 2),
    (2, 0)
]

U, edge_to_index = build_eulerian_walk_operator(edges)

print("U =")
print(U)

print("Is U unitary?")
print(np.allclose(U.conj().T @ U, np.eye(len(edges))))

def simulate_walk(U, start_edge_index, T):
    """
    Simulate psi(t+1) = U psi(t).
    """
    n = U.shape[0]

    psi = np.zeros(n, dtype=complex)
    psi[start_edge_index] = 1.0

    states = []

    for t in range(T + 1):
        states.append(psi.copy())
        psi = U @ psi

    return np.array(states)

T = 20
start_edge = (0, 1)
start_index = edge_to_index[start_edge]

states = simulate_walk(U, start_index, T)

probabilities = np.abs(states) ** 2

print(probabilities)

def plot_edge_probabilities(probabilities, edges):
    T = probabilities.shape[0]

    plt.figure(figsize=(10, 5))

    for i, e in enumerate(edges):
        plt.plot(range(T), probabilities[:, i], label=str(e))

    plt.xlabel("Time step")
    plt.ylabel("Probability")
    plt.title("Edge probabilities over time")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_edge_probabilities(probabilities, edges)

edges2 = [
    (0, 1),
    (1, 2),
    (2, 0),

    (1, 3),
    (3, 4),
    (4, 1)
]

print(check_eulerian(edges2))

U2, edge_to_index2 = build_eulerian_walk_operator(edges2)

print("Is U2 unitary?")
print(np.allclose(U2.conj().T @ U2, np.eye(len(edges2))))

T = 30
start_edge = (0, 1)
start_index = edge_to_index2[start_edge]

states2 = simulate_walk(U2, start_index, T)
probabilities2 = np.abs(states2) ** 2

plot_edge_probabilities(probabilities2, edges2)

def return_probability(probabilities, start_edge_index):
    return probabilities[:, start_edge_index]


def plot_return_probability(probabilities, start_edge_index):
    P_return = return_probability(probabilities, start_edge_index)

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(P_return)), P_return)
    plt.xlabel("Time step")
    plt.ylabel("Return probability")
    plt.title("Return probability")
    plt.grid(True)
    plt.show()

plot_return_probability(probabilities2, start_index)

def long_time_average(probabilities):
    return np.mean(probabilities, axis=0)


def plot_long_time_average(probabilities, edges):
    avg = long_time_average(probabilities)

    labels = [str(e) for e in edges]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, avg)
    plt.xlabel("Edge")
    plt.ylabel("Long-time average probability")
    plt.title("Long-time average edge occupation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_long_time_average(probabilities2, edges2)

def plot_spectrum(U):
    eigvals = np.linalg.eigvals(U)

    plt.figure(figsize=(5, 5))
    plt.scatter(eigvals.real, eigvals.imag)

    circle = plt.Circle((0, 0), 1, fill=False)
    plt.gca().add_artist(circle)

    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Eigenvalues of U")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    return eigvals

eigvals2 = plot_spectrum(U2)
print(eigvals2)
