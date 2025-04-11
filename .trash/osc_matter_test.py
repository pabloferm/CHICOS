import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm  # Matrix exponential

# Mixing angles (radians)
theta_12 = np.radians(33.44)
theta_23 = np.radians(49.2)
theta_13 = np.radians(8.57)
delta_cp = np.radians(234)  # CP-violating phase

# Mass-squared differences (eV²)
dm2_21 = 7.42e-5
dm2_31 = 2.514e-3


# Define PMNS matrix
def pmns_matrix(theta_12, theta_23, theta_13, delta_cp):
    c12, s12 = np.cos(theta_12), np.sin(theta_12)
    c23, s23 = np.cos(theta_23), np.sin(theta_23)
    c13, s13 = np.cos(theta_13), np.sin(theta_13)

    U = np.array(
        [
            [c12 * c13, s12 * c13, s13 * np.exp(-1j * delta_cp)],
            [
                -s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta_cp),
                c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta_cp),
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta_cp),
                -c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta_cp),
                c23 * c13,
            ],
        ]
    )
    return U


U_PMNS = pmns_matrix(theta_12, theta_23, theta_13, delta_cp)

# Matter effects
G_F = 1.1663787e-5  # Fermi constant in eV⁻²
density = 2.8  # Earth's crust density in g/cm³
Y_e = 0.5  # Electron fraction
N_A = 6.022e23  # Avogadro's number
m_p = 1.6726e-24  # Proton mass in g

# Compute electron number density
N_e = density * Y_e * N_A / m_p  # electrons/cm³
V_e = np.sqrt(2) * G_F * N_e * 1e-6  # Convert to eV


# Define vacuum Hamiltonian
def hamiltonian_vacuum(U, dm2, E):
    M2 = np.diag([0, dm2[0], dm2[1]])  # Mass-squared matrix
    return U @ M2 @ np.conj(U).T / (2 * E)


# Compute the evolution operator without eigenvalues
def oscillation_probabilities_matter(U, dm2, L, E, V_e):
    H_vac = hamiltonian_vacuum(U, dm2, E)  # Vacuum Hamiltonian
    V_matrix = np.array([[V_e, 0, 0], [0, 0, 0], [0, 0, 0]])  # Matter potential

    H_matter = H_vac + V_matrix  # Total Hamiltonian in matter
    P = {}  # Store transition probabilities

    for i, L_i in enumerate(L):
        U_flavor = expm(-1j * H_matter * L_i)  # Evolution operator

        for alpha in range(3):  # Initial flavors
            for beta in range(3):  # Final flavors
                P[(alpha, beta, i)] = (
                    np.abs(U_flavor[beta, alpha]) ** 2
                )  # Probability at L_i

    return P


# Define baseline range and neutrino energy
L = np.linspace(0, 1000, 1000)  # Distance in km
E = 1.0  # Neutrino energy in GeV

# Compute probabilities in matter
dm2_values = [dm2_21, dm2_31]
P_matter = oscillation_probabilities_matter(U_PMNS, dm2_values, L, E, V_e)

# Define flavor labels
flavors = {0: "e", 1: "μ", 2: "τ"}

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# **Plot 1: Survival Probabilities**
for alpha in range(3):
    axs[0].plot(
        L,
        [P_matter[(alpha, alpha, i)] for i in range(len(L))],
        label=rf"$P(\nu_{flavors[alpha]} \to \nu_{flavors[alpha]})$",
    )
axs[0].set_ylabel("Survival Probability")
axs[0].set_title("Neutrino Survival Probabilities in Matter")
axs[0].legend()
axs[0].grid()

# **Plot 2: Electron Neutrino Appearance**
for alpha in [1, 2]:  # Muon and Tau to Electron transitions
    axs[1].plot(
        L,
        [P_matter[(alpha, 0, i)] for i in range(len(L))],
        label=rf"$P(\nu_{flavors[alpha]} \to \nu_e)$",
    )
axs[1].set_ylabel("Appearance Probability")
axs[1].set_title("Electron Neutrino Appearance in Matter")
axs[1].legend()
axs[1].grid()

# **Plot 3: Muon-Tau Transitions**
axs[2].plot(
    L,
    [P_matter[(1, 2, i)] for i in range(len(L))],
    label=r"$P(\nu_\mu \to \nu_\tau)$",
    color="purple",
)
axs[2].set_xlabel("Baseline Distance L (km)")
axs[2].set_ylabel("Transition Probability")
axs[2].set_title("Muon to Tau Neutrino Oscillation in Matter")
axs[2].legend()
axs[2].grid()

# Show the full figure
plt.tight_layout()
plt.show()
