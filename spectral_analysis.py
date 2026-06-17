import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

from simple_random_walk import U_grover, U_random, grover_coin, build_eulerian_walk_operator

eigvals, eigvecs = np.linalg.eig(U_grover)

def plot_eigenphases(U, title="Eigenphases"):
    eigvals = np.linalg.eigvals(U)
    phases = np.angle(eigvals)

    plt.figure(figsize=(8, 4))
    plt.scatter(range(len(phases)), np.sort(phases))
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Phase")
    plt.title(title)
    plt.grid(True)
    plt.show()

    return phases


phases_grover = plot_eigenphases(U_grover, "Eulerian Grover eigenphases")
phases_random = plot_eigenphases(U_random, "Random unitary eigenphases")


def print_eigenvalue_degeneracies(U, tol=1e-8):
    eigvals = np.linalg.eigvals(U)
    phases = np.sort(np.angle(eigvals))

    groups = []
    current_group = [phases[0]]

    for p in phases[1:]:
        if abs(p - current_group[-1]) < tol:
            current_group.append(p)
        else:
            groups.append(current_group)
            current_group = [p]

    groups.append(current_group)

    for g in groups:
        if len(g) > 1:
            print(f"Phase {g[0]:.6f} has degeneracy {len(g)}")

print("Grover degeneracies:")
print_eigenvalue_degeneracies(U_grover)

print("Random degeneracies:")
print_eigenvalue_degeneracies(U_random)

def eigenvector_ipr(U):
    eigvals, eigvecs = np.linalg.eig(U)

    iprs = []

    for k in range(eigvecs.shape[1]):
        phi = eigvecs[:, k]
        p = np.abs(phi) ** 2
        p = p / np.sum(p)
        iprs.append(np.sum(p ** 2))

    return eigvals, np.array(iprs)


eigvals_g, ipr_g = eigenvector_ipr(U_grover)
eigvals_r, ipr_r = eigenvector_ipr(U_random)

plt.figure(figsize=(8, 4))
plt.scatter(np.angle(eigvals_g), ipr_g, label="Eulerian Grover")
plt.scatter(np.angle(eigvals_r), ipr_r, label="Random unitary")
plt.xlabel("Eigenphase")
plt.ylabel("Eigenvector IPR")
plt.title("Eigenmode localization")
plt.legend()
plt.grid(True)
plt.show()

def plot_most_localized_eigenvector(U, edges, title="Most localized eigenvector"):
    eigvals, eigvecs = np.linalg.eig(U)

    iprs = []
    probs = []

    for k in range(eigvecs.shape[1]):
        phi = eigvecs[:, k]
        p = np.abs(phi) ** 2
        p = p / np.sum(p)
        probs.append(p)
        iprs.append(np.sum(p ** 2))

    kmax = np.argmax(iprs)

    plt.figure(figsize=(10, 4))
    plt.bar([str(e) for e in edges], probs[kmax])
    plt.xticks(rotation=45)
    plt.ylabel("Probability")
    plt.title(title + f"\nEigenphase = {np.angle(eigvals[kmax]):.3f}, IPR = {iprs[kmax]:.3f}")
    plt.tight_layout()
    plt.show()

    return eigvals[kmax], eigvecs[:, kmax], iprs[kmax]

edges = [
    (0, 1),
    (1, 2),
    (2, 0),

    (1, 3),
    (3, 4),
    (4, 1)
]

lam, phi, ipr = plot_most_localized_eigenvector(
    U_grover, edges, "Most localized Eulerian Grover eigenmode"
)

def eigenmode_overlaps(U, start_index):
    eigvals, eigvecs = np.linalg.eig(U)

    psi0 = np.zeros(U.shape[0], dtype=complex)
    psi0[start_index] = 1.0

    overlaps = []

    for k in range(eigvecs.shape[1]):
        phi = eigvecs[:, k]
        phi = phi / np.linalg.norm(phi)
        c = np.vdot(phi, psi0)
        overlaps.append(abs(c) ** 2)

    return eigvals, np.array(overlaps)

eigvals, overlaps = eigenmode_overlaps(U_grover, start_index)

plt.figure(figsize=(8, 4))
plt.scatter(np.angle(eigvals), overlaps)
plt.xlabel("Eigenphase")
plt.ylabel("Overlap with initial state")
plt.title("Initial-state overlap with eigenmodes")
plt.grid(True)
plt.show()

def compare_spectral_properties(U1, U2, labels=("Grover", "Random")):
    eig1, ipr1 = eigenvector_ipr(U1)
    eig2, ipr2 = eigenvector_ipr(U2)

    plt.figure(figsize=(8, 4))
    plt.scatter(np.angle(eig1), ipr1, label=labels[0])
    plt.scatter(np.angle(eig2), ipr2, label=labels[1])
    plt.xlabel("Eigenphase")
    plt.ylabel("Eigenvector IPR")
    plt.title("Spectral localization comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

compare_spectral_properties(U_grover, U_random)


