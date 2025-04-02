import numpy as np
import matplotlib.pyplot as plt

# Constants
G_F = 1.1663787e-5  # Fermi coupling constant in GeV^-2
n_e = 1e6  # Example electron number density (in cm^-3)
epsilon_params = {
    "ee": 0.01,
    "emu": 0.01,
    "etau": 0.01,
    "mumu": 0.01,
    "mutau": 0.01,
    "tautau": 0.01,
}
xi = 1e-7  # Lorentz violation parameter
gamma = 1e-6  # Decoherence parameter
delta_m2_21 = 7.53e-5  # Solar mass squared difference (eV^2)
delta_m2_31 = 2.44e-3  # Atmospheric mass squared difference (eV^2)

# Logarithmic energy range
energy_range = np.logspace(-3, 0, 100)  # From 0.1 GeV to 100 GeV


# Function for shifted Hamiltonian
def shift_hamiltonian(H):
    trace_H = np.trace(H)
    H_shift = H - (trace_H / 3) * np.identity(3)
    return H_shift


# Functions for Hamiltonian components
def h_vacuum(E):
    return np.array(
        [[0, 0, 0], [0, delta_m2_21 / (2 * E), 0], [0, 0, delta_m2_31 / (2 * E)]]
    )


def h_matter(E):
    return np.array([[np.sqrt(2) * G_F * n_e, 0, 0], [0, 0, 0], [0, 0, 0]])


def h_nsi(E):
    return np.array(
        [
            [
                epsilon_params["ee"] * G_F * n_e,
                epsilon_params["emu"] * G_F * n_e,
                epsilon_params["etau"] * G_F * n_e,
            ],
            [
                epsilon_params["emu"] * G_F * n_e,
                epsilon_params["mumu"] * G_F * n_e,
                epsilon_params["mutau"] * G_F * n_e,
            ],
            [
                epsilon_params["etau"] * G_F * n_e,
                epsilon_params["mutau"] * G_F * n_e,
                epsilon_params["tautau"] * G_F * n_e,
            ],
        ]
    )


def h_lorentz(E):
    return xi * E * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])


def h_decoherence(E):
    return np.diag([0, gamma, gamma])


# Function to compute determinant and trace ratio
def compute_ratio(H):
    H_shift = shift_hamiltonian(H)
    det_H_shift = np.linalg.det(H_shift)
    trace_H_shift_sq = np.trace(np.dot(H_shift, H_shift))
    return det_H_shift / trace_H_shift_sq if trace_H_shift_sq != 0 else 0


def compute_trace(H):
    return np.trace(H)


def compute_det(H):
    return np.linalg.det(H)


# Compute ratios for each effect
ratios = {
    # "Trace of the Hamiltonian": [],
    # "Determinant of the Hamiltonian": [],
    # "Trace of the shifted Hamiltonian squared": [],
    # "Determinant of the shifted Hamiltonian": [],
    # "Approx. of determinant of the shifted Hamiltonian": [],
    "theta_H/3": [],
    "theta_H/3 + 2pi/3": [],
    "theta_H/3 + 4pi/3": [],
    # "NSI": [],
    # "Lorentz": [],
    # "Decoherence": []
}
for E in energy_range:
    Hs = shift_hamiltonian(h_vacuum(E) + h_matter(E))
    H = h_vacuum(E) + h_matter(E)
    trH = compute_trace(H) / 3
    trH2 = compute_trace(np.dot(H, H))
    b = -0.5 * compute_trace(np.dot(Hs, Hs))
    scale = np.sqrt(2 * compute_trace(np.dot(Hs, Hs)) / 3)
    scale = 1.0
    # ratios["Trace of the Hamiltonian"].append(compute_trace(h_vacuum(E) + h_matter(E)))
    # ratios["Determinant of the Hamiltonian"].append(
    #     compute_det(h_matter(E) + h_vacuum(E))
    # )
    # ratios["Trace of the shifted Hamiltonian squared"].append(
    #     compute_trace(np.dot(Hs, Hs))
    # )
    # ratios["theta_H/3"].append(
    #     scale
    #     * np.cos(
    #         np.arccos(
    #             np.sqrt(27 * compute_det(Hs) ** 2 / compute_trace(np.dot(Hs, Hs)) ** 3)
    #         )
    #         / 3
    #     )
    # )
    # ratios["theta_H/3 + 2pi/3"].append(
    #     scale
    #     * np.cos(
    #         np.arccos(
    #             np.sqrt(27 * compute_det(Hs) ** 2 / compute_trace(np.dot(Hs, Hs)) ** 3)
    #         )
    #         / 3
    #         + 2 * np.pi / 3
    #     )
    #     + trH
    # )
    # ratios["theta_H/3 + 4pi/3"].append(
    #     scale
    #     * np.cos(
    #         np.arccos(
    #             np.sqrt(27 * compute_det(Hs) ** 2 / compute_trace(np.dot(Hs, Hs)) ** 3)
    #         )
    #         / 3
    #         + 4 * np.pi / 3
    #     )
    #     + trH
    # )
    ratios["TrH2 direct"].append(trH2)
    ratios["TrH2 formula"].append(delta_m2_21**2 + delta_m2_31**2) / (4 * E**2)
            + (np.sqrt(2) * G_F * n_e)**2
            + (np.sqrt(2) * G_F * n_e)*(
                self.U[0, 1] * np.conj(self.U[0, 1]) * self.dm2_21
                + self.U[1, 2] * np.conj(self.U[1, 2]) * self.dm2_31
            )
            / E
    )

# Plot results
plt.figure(figsize=(10, 6))

for key, values in ratios.items():
    plt.plot(energy_range, values, label=key)

plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Energy (GeV)")
plt.legend()
plt.grid(True)
plt.show()
