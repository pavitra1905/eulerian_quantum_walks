import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path("phase4_outputs")


def coin(theta):
    return np.array(
        [
            [np.cos(theta), np.sin(theta)],
            [np.sin(theta), -np.cos(theta)],
        ],
        dtype=complex,
    )


def shift_k(k):
    return np.array(
        [
            [np.exp(1j * k), 0],
            [0, np.exp(-1j * k)],
        ],
        dtype=complex,
    )


def U_k(k, theta):
    return shift_k(k) @ coin(theta)


def compute_bands(theta, n_k=400):
    ks = np.linspace(-np.pi, np.pi, n_k)
    bands = []

    for k in ks:
        eigvals = np.linalg.eigvals(U_k(k, theta))
        bands.append(np.sort(np.angle(eigvals)))

    return ks, np.array(bands)


def band_windings(bands):
    windings = []

    for j in range(bands.shape[1]):
        phase = np.unwrap(bands[:, j])
        windings.append((phase[-1] - phase[0]) / (2 * np.pi))

    return np.array(windings)


def determinant_winding(theta, n_k=400):
    ks = np.linspace(-np.pi, np.pi, n_k)
    phases = np.unwrap([np.angle(np.linalg.det(U_k(k, theta))) for k in ks])
    return (phases[-1] - phases[0]) / (2 * np.pi)


def plot_bands(theta_values):
    plt.figure(figsize=(8, 4))

    for theta in theta_values:
        ks, bands = compute_bands(theta)
        for j in range(bands.shape[1]):
            label = f"theta={theta:.2f}" if j == 0 else None
            plt.plot(ks, bands[:, j], label=label)

    plt.xlabel("Bloch momentum k")
    plt.ylabel("Eigenphase")
    plt.title("Periodic Eulerian walk quasienergy bands")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase4_quasienergy_bands.png", dpi=200)
    plt.close()


def build_interface_operator(n_cells, theta_left, theta_right):
    dim = 2 * n_cells
    C_total = np.zeros((dim, dim), dtype=complex)
    S = np.zeros((dim, dim), dtype=complex)

    for n in range(n_cells):
        theta = theta_left if n < n_cells // 2 else theta_right
        C_total[2 * n : 2 * n + 2, 2 * n : 2 * n + 2] = coin(theta)

    for n in range(n_cells):
        r = 2 * n
        l = 2 * n + 1

        if n + 1 < n_cells:
            S[2 * (n + 1), r] = 1.0
        else:
            S[l, r] = 1.0

        if n - 1 >= 0:
            S[2 * (n - 1) + 1, l] = 1.0
        else:
            S[r, l] = 1.0

    return S @ C_total


def position_probabilities(eigvec, n_cells):
    probs = np.abs(eigvec) ** 2
    probs = probs / np.sum(probs)
    return probs.reshape(n_cells, 2).sum(axis=1)


def interface_scores(U, n_cells):
    eigvals, eigvecs = np.linalg.eig(U)
    center = (n_cells - 1) / 2
    xs = np.arange(n_cells)
    rows = []

    for k in range(eigvecs.shape[1]):
        p = position_probabilities(eigvecs[:, k], n_cells)
        ipr = np.sum(p**2)
        mean_distance = np.sum(np.abs(xs - center) * p)
        score = ipr / (1.0 + mean_distance)
        rows.append((score, k, np.angle(eigvals[k]), ipr, mean_distance, p))

    return sorted(rows, reverse=True)


def plot_interface_modes(theta_left, theta_right, n_cells=60, top_n=3):
    U = build_interface_operator(n_cells, theta_left, theta_right)
    rows = interface_scores(U, n_cells)

    plt.figure(figsize=(9, 4))
    for rank, (_, k, phase, ipr, _, p) in enumerate(rows[:top_n], start=1):
        plt.plot(
            np.arange(n_cells),
            p,
            label=f"mode {k}, phase={phase:.2f}, IPR={ipr:.3f}",
        )

    plt.axvline(n_cells // 2 - 0.5, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Unit cell")
    plt.ylabel("Probability")
    plt.title(
        f"Candidate interface-localized modes, theta_L={theta_left:.2f}, theta_R={theta_right:.2f}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase4_interface_modes.png", dpi=200)
    plt.close()

    return rows


def grover_coin(d):
    if d == 1:
        return np.array([[1.0]], dtype=complex)
    return (2 / d) * np.ones((d, d), dtype=complex) - np.eye(d, dtype=complex)


def build_eulerian_walk_operator(edges):
    vertices = sorted(set([v for edge in edges for v in edge]))
    incoming = {v: [] for v in vertices}
    outgoing = {v: [] for v in vertices}
    edge_to_index = {e: i for i, e in enumerate(edges)}

    for e in edges:
        u, v = e
        outgoing[u].append(e)
        incoming[v].append(e)

    U = np.zeros((len(edges), len(edges)), dtype=complex)
    for v in vertices:
        if len(incoming[v]) != len(outgoing[v]):
            raise ValueError("Graph is not Eulerian.")

        C = grover_coin(len(incoming[v]))
        for i, e_in in enumerate(incoming[v]):
            for j, e_out in enumerate(outgoing[v]):
                U[edge_to_index[e_out], edge_to_index[e_in]] = C[j, i]

    return U


def plot_edges3_spectral_reference():
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
    U = build_eulerian_walk_operator(edges3)
    eigvals, eigvecs = np.linalg.eig(U)
    iprs = []

    for k in range(eigvecs.shape[1]):
        p = np.abs(eigvecs[:, k]) ** 2
        p = p / np.sum(p)
        iprs.append(np.sum(p**2))

    plt.figure(figsize=(8, 4))
    plt.scatter(np.angle(eigvals), iprs)
    plt.xlabel("Eigenphase")
    plt.ylabel("Eigenvector IPR")
    plt.title("Finite edges3 spectral localization reference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "edges3_spectral_localization_reference.png", dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    theta_values = [-np.pi / 3, np.pi / 3]
    plot_bands(theta_values)

    print("Periodic model winding diagnostics")
    for theta in theta_values:
        ks, bands = compute_bands(theta)
        print(f"theta={theta:.4f}")
        print("  band windings:", np.round(band_windings(bands), 6))
        print("  determinant winding:", round(determinant_winding(theta), 6))

    rows = plot_interface_modes(theta_left=np.pi / 3, theta_right=-np.pi / 3)
    print("\nTop candidate interface-localized modes")
    print("rank | mode | eigenphase | position IPR | mean distance from interface")
    for rank, (_, k, phase, ipr, mean_distance, _) in enumerate(rows[:8], start=1):
        print(f"{rank:4d} | {k:4d} | {phase:10.4f} | {ipr:12.4f} | {mean_distance:28.4f}")

    plot_edges3_spectral_reference()
    print("\nSaved plots to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
