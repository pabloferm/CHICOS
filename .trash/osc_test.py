import numpy as np
import matplotlib.pyplot as plt

# Define the standard PMNS mixing angles (converted to radians)
theta_12 = np.radians(33.44)
theta_23 = np.radians(49.2)
theta_13 = np.radians(8.57)
delta_cp = np.radians(234)  # CP-violating phase

# Define mass-squared differences (in eV²)
dm2_21 = 7.42e-5  # Solar mass-squared difference
dm2_31 = 2.514e-3  # Atmospheric mass-squared difference
dm2_32 = dm2_31 - dm2_21  # Third mass-squared difference


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


# Define function to compute oscillation probabilities
def oscillation_probabilities(U, dm2, L, E):
    """
    Compute P(ν_alpha → ν_beta) for all (alpha, beta) given the PMNS matrix and mass-squared differences.

    Arguments:
    - U: PMNS matrix
    - dm2: [dm2_21, dm2_31, dm2_32]
    - L: array of baseline distances (km)
    - E: neutrino energy (GeV)

    Returns:
    - P: Dictionary of probabilities for all flavor transitions
    """
    alpha_factor = 1.267 * L / E  # Conversion factor for oscillation phase
    P = {}  # Store all transition probabilities

    for alpha in range(3):  # Initial flavors: 0=e, 1=μ, 2=τ
        for beta in range(3):  # Final flavors
            sum_terms = np.zeros_like(L, dtype=complex)
            for i in range(3):  # Sum over mass eigenstates
                phase = np.exp(-1j * alpha_factor * dm2[i])
                sum_terms += U[beta, i] * np.conj(U[alpha, i]) * phase

            P[(alpha, beta)] = np.abs(sum_terms) ** 2  # Compute probability

    return P


# Define baseline range and neutrino energy
L = np.geomspace(0.1, 10000, 1000)  # Baseline distance in km
E = 1.0  # Neutrino energy in GeV

# Compute all oscillation probabilities
dm2_values = [dm2_21, dm2_31, dm2_32]
P_all = oscillation_probabilities(U_PMNS, dm2_values, L, E)

# Define flavor labels
flavors = {0: "e", 1: "μ", 2: "τ"}

# Plot all oscillation probabilities
plt.figure(figsize=(10, 6))

for (alpha, beta), P in P_all.items():
    plt.plot(L, P, label=rf"$P(\nu_{flavors[alpha]} \to \nu_{flavors[beta]})$")

plt.xlabel("Baseline Distance L (km)")
plt.ylabel("Oscillation Probability")
plt.title("Three-Flavor Neutrino Oscillation Probabilities in Vacuum")
plt.legend()
plt.grid()
plt.show()
