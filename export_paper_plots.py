from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from phase4_topology import (
    build_interface_operator,
    compute_bands,
    interface_scores,
)


OUTPUT_DIR = Path("paper_figures")


def grover_coin(d):
    if d == 1:
        return np.array([[1.0]], dtype=complex)

    return (2 / d) * np.ones((d, d), dtype=complex) - np.eye(d, dtype=complex)


def random_unitary(d):
    z = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    q, r = np.linalg.qr(z)
    phases = np.diag(r) / np.abs(np.diag(r))
    return q * phases


def check_eulerian(edges):
    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    vertices = set()

    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1
        vertices.add(u)
        vertices.add(v)

    return all(indeg[v] == outdeg[v] for v in vertices)


def build_walk_operator(edges, coin_type="grover", seed=None):
    if seed is not None:
        np.random.seed(seed)

    if not check_eulerian(edges):
        raise ValueError("Graph is not Eulerian.")

    edge_to_index = {e: i for i, e in enumerate(edges)}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    vertices = set()

    for e in edges:
        u, v = e
        outgoing[u].append(e)
        incoming[v].append(e)
        vertices.add(u)
        vertices.add(v)

    u_matrix = np.zeros((len(edges), len(edges)), dtype=complex)

    for v in vertices:
        d = len(incoming[v])
        if coin_type == "grover":
            coin = grover_coin(d)
        elif coin_type == "random":
            coin = random_unitary(d)
        else:
            raise ValueError("coin_type must be 'grover' or 'random'.")

        for i, e_in in enumerate(incoming[v]):
            for j, e_out in enumerate(outgoing[v]):
                row = edge_to_index[e_out]
                col = edge_to_index[e_in]
                u_matrix[row, col] = coin[j, i]

    return u_matrix, edge_to_index


def simulate_walk(u_matrix, start_index, steps):
    psi = np.zeros(u_matrix.shape[0], dtype=complex)
    psi[start_index] = 1.0
    states = []

    for _ in range(steps + 1):
        states.append(psi.copy())
        psi = u_matrix @ psi

    return np.array(states)


def compute_ipr(probabilities):
    return np.sum(probabilities**2, axis=1)


def eigenvector_ipr(u_matrix):
    eigvals, eigvecs = np.linalg.eig(u_matrix)
    iprs = []

    for k in range(eigvecs.shape[1]):
        probs = np.abs(eigvecs[:, k]) ** 2
        probs = probs / np.sum(probs)
        iprs.append(np.sum(probs**2))

    return eigvals, np.array(iprs)


def save_current(name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_return_probability(probs_grover, probs_random, start_index):
    plt.figure(figsize=(7, 4))
    plt.plot(probs_grover[:, start_index], label="Eulerian Grover")
    plt.plot(probs_random[:, start_index], label="Random local unitary")
    plt.xlabel("Time step")
    plt.ylabel("Return probability")
    plt.title("Return probability comparison")
    plt.legend()
    plt.grid(True)
    save_current("return_probability_comparison")


def plot_localization(probs_grover, probs_random):
    plt.figure(figsize=(7, 4))
    plt.plot(compute_ipr(probs_grover), label="Eulerian Grover")
    plt.plot(compute_ipr(probs_random), label="Random local unitary")
    plt.xlabel("Time step")
    plt.ylabel("IPR")
    plt.title("Localization comparison")
    plt.legend()
    plt.grid(True)
    save_current("localization_ipr_comparison")


def plot_spectrum(u_grover, u_random):
    eig_grover = np.linalg.eigvals(u_grover)
    eig_random = np.linalg.eigvals(u_random)

    plt.figure(figsize=(5, 5))
    plt.scatter(eig_grover.real, eig_grover.imag, label="Eulerian Grover")
    plt.scatter(eig_random.real, eig_random.imag, label="Random local unitary")
    plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, color="black"))
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Spectrum comparison")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    save_current("spectrum_comparison")


def plot_combined_eigenphases(u_grover, u_random):
    phases_grover = np.sort(np.angle(np.linalg.eigvals(u_grover)))
    phases_random = np.sort(np.angle(np.linalg.eigvals(u_random)))

    plt.figure(figsize=(7, 4))
    plt.scatter(range(len(phases_grover)), phases_grover, label="Eulerian Grover")
    plt.scatter(range(len(phases_random)), phases_random, label="Random local unitary")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenphase")
    plt.title("Eigenphase comparison")
    plt.legend()
    plt.grid(True)
    save_current("eigenphase_comparison")


def plot_spectral_localization(u_grover, u_random):
    eig_grover, ipr_grover = eigenvector_ipr(u_grover)
    eig_random, ipr_random = eigenvector_ipr(u_random)

    plt.figure(figsize=(7, 4))
    plt.scatter(np.angle(eig_grover), ipr_grover, label="Eulerian Grover")
    plt.scatter(np.angle(eig_random), ipr_random, label="Random local unitary")
    plt.xlabel("Eigenphase")
    plt.ylabel("Eigenvector IPR")
    plt.title("Spectral localization comparison")
    plt.legend()
    plt.grid(True)
    save_current("spectral_localization_comparison")


def plot_phase4_bands():
    theta_values = [-np.pi / 3, np.pi / 3]
    plt.figure(figsize=(7, 4))

    for theta in theta_values:
        ks, bands = compute_bands(theta)
        for j in range(bands.shape[1]):
            label = rf"$\theta={theta:.2f}$" if j == 0 else None
            plt.plot(ks, bands[:, j], label=label)

    plt.xlabel("Bloch momentum $k$")
    plt.ylabel("Eigenphase")
    plt.title("Periodic Eulerian walk quasienergy bands")
    plt.legend()
    plt.grid(True)
    save_current("phase4_quasienergy_bands")


def plot_phase4_interface_modes():
    theta_left = np.pi / 3
    theta_right = -np.pi / 3
    n_cells = 60
    top_n = 3
    u_matrix = build_interface_operator(n_cells, theta_left, theta_right)
    rows = interface_scores(u_matrix, n_cells)

    plt.figure(figsize=(7, 4))
    for _, k, phase, ipr, _, probs in rows[:top_n]:
        plt.plot(
            np.arange(n_cells),
            probs,
            label=rf"mode {k}, phase={phase:.2f}, IPR={ipr:.3f}",
        )

    plt.axvline(n_cells // 2 - 0.5, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Unit cell")
    plt.ylabel("Probability")
    plt.title(rf"Interface modes, $\theta_L={theta_left:.2f}$, $\theta_R={theta_right:.2f}$")
    plt.legend()
    plt.grid(True)
    save_current("phase4_interface_modes")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    edges3 = [
        (0, 1),
        (1, 2),
        (2, 0),
        (1, 3),
        (3, 4),
        (4, 1),
        (1, 5),
        (5, 6),
        (6, 1),
        (0, 7),
        (7, 0),
        (7, 2),
        (2, 8),
        (8, 7),
    ]

    u_grover, edge_to_index = build_walk_operator(edges3, coin_type="grover")
    u_random, _ = build_walk_operator(edges3, coin_type="random", seed=1)

    start_index = edge_to_index[(0, 1)]
    steps = 50
    probs_grover = np.abs(simulate_walk(u_grover, start_index, steps)) ** 2
    probs_random = np.abs(simulate_walk(u_random, start_index, steps)) ** 2

    plot_return_probability(probs_grover, probs_random, start_index)
    plot_localization(probs_grover, probs_random)
    plot_spectrum(u_grover, u_random)
    plot_combined_eigenphases(u_grover, u_random)
    plot_spectral_localization(u_grover, u_random)
    plot_phase4_bands()
    plot_phase4_interface_modes()

    print(f"Saved PDF and PNG figures to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
