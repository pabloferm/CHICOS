import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


class NuMixingMatrices:
    def __init__(self, theta_23, theta_13, theta_12, delta_cp):
        c23 = np.cos(theta_23)
        s23 = np.sin(theta_23)
        c13 = np.cos(theta_13)
        s13 = np.sin(theta_13)
        c12 = np.cos(theta_12)
        s12 = np.sin(theta_12)
        dcp = delta_cp

        R12 = self.R12(c12, s12)
        R23 = self.R23(c23, s23)
        R13d = self.R13d(c13, s13, dcp)

        d12_R12 = self.d12_R12(c12, s12)
        d23_R23 = self.d23_R23(c23, s23)
        d13_R13d = self.d13_R13d(c13, s13, dcp)
        dd_R13d = self.dd_R13d(c13, s13, dcp)

        self.U = R23 @ R13d @ R12
        self.ccU = np.asarray(np.asmatrix(self.U).H)
        self.cU = np.conjugate(self.U)
        self.d23_U = d23_R23 @ R13d @ R12
        self.d12_U = R23 @ R13d @ d12_R12
        self.d13_U = R23 @ d13_R13d @ R12
        self.dd_U = R23 @ dd_R13d @ R12

    def R12(self, c12, s12):
        return np.array([[c12, s12, 0], [-s12, c12, 0], [0, 0, 1]])

    def d12_R12(self, c12, s12):
        return np.array([[-s12, c12, 0], [-c12, -s12, 0], [0, 0, 0]])

    def R23(self, c23, s23):
        return np.array(
            [
                [1, 0, 0],
                [0, c23, s23],
                [0, -s23, c23],
            ]
        )

    def d23_R23(self, c23, s23):
        return np.array(
            [
                [0, 0, 0],
                [0, -s23, c23],
                [0, -c23, -s23],
            ]
        )

    def R13d(self, c13, s13, dcp):
        return np.array(
            [
                [c13, 0, s13 * np.exp(-dcp * 1j)],
                [0, 1, 0],
                [-s13 * np.exp(dcp * 1j), 0, c13],
            ]
        )

    def dd_R13d(self, c13, s13, dcp):
        return np.array(
            [
                [0, 0, -1j * s13 * np.exp(-dcp * 1j)],
                [0, 0, 0],
                [-1j * s13 * np.exp(dcp * 1j), 0, 0],
            ]
        )

    def d13_R13d(self, c13, s13, dcp):
        return np.array(
            [
                [-s13, 0, c13 * np.exp(-dcp * 1j)],
                [0, 0, 0],
                [-c13 * np.exp(dcp * 1j), 0, -s13],
            ]
        )

    def print_matrices(self):
        self._print_matrix("PMNS", self.U)
        self._print_matrix("PMNS Hermitian conjugate", self.ccU)
        self._print_matrix("PMNS element conjugate", self.cU)
        self._print_matrix("Derivative of PMNS w.r.t. theta_23", self.d23_U)
        self._print_matrix("Derivative of PMNS w.r.t. theta_12", self.d12_U)
        self._print_matrix("Derivative of PMNS w.r.t. theta_13", self.d13_U)
        self._print_matrix("Derivative of PMNS w.r.t. delta_cp", self.dd_U)

    def _print_matrix(self, name, matrix, decimals=2):
        print(f"Printing {name} matrix:")
        print(np.around(matrix, decimals=decimals))


class Probability(NuMixingMatrices):
    """docstring for Probability"""

    def __init__(self, theta_23, theta_13, theta_12, delta_cp, Delta_m221, Delta_m231):
        super(Probability, self).__init__(theta_23, theta_13, theta_12, delta_cp)

        self.M2 = np.diag([0, Delta_m221, Delta_m231])
        self.dm2 = [Delta_m221, Delta_m231, Delta_m231 - Delta_m221]

        # Matter effects
        G_F = 1.1663787e-5  # Fermi constant in eV⁻²
        density = 2.8  # Earth's crust density in g/cm³
        Y_e = 0.5  # Electron fraction
        N_A = 6.022e23  # Avogadro's number
        m_p = 1.6726e-24  # Proton mass in g

        # Compute electron number density
        N_e = density * Y_e * N_A / m_p  # electrons/cm³
        V_e = np.sqrt(2) * G_F * N_e * 1e-6  # Convert to eV
        self.V_matrix = np.array(
            [[V_e, 0, 0], [0, 0, 0], [0, 0, 0]]
        )  # Matter potential

    def hamiltonian_vacuum(self, E):
        return self.U @ self.M2 @ np.conj(self.U).T / (2 * E)

    def value(self, L_values, E_values):
        P_matrix = np.zeros((len(L_values), len(E_values), 3, 3))  # Shape: (L, E, α, β)

        for i, L in enumerate(L_values):
            for j, E in enumerate(E_values):
                H_vac = self.hamiltonian_vacuum(E)  # Vacuum Hamiltonian
                H = H_vac + self.V_matrix  # Total Hamiltonian in matter
                U_flavor = expm(-1j * H * L)  # Evolution operator

                for α in range(3):
                    for β in range(3):
                        P_matrix[i, j, α, β] = (
                            np.abs(U_flavor[β, α]) ** 2
                        )  # Probabilities
        return P_matrix

    def dP_d23(self, L, E):
        alpha_factor = 1.267 * L / E  # Conversion factor for oscillation phase
        P = {}  # Store all transition probabilities

        for alpha in range(3):  # Initial flavors: 0=e, 1=μ, 2=τ
            for beta in range(3):  # Final flavors
                sum_terms = np.zeros_like(alpha_factor, dtype=complex)
                d_sum_terms = np.zeros_like(alpha_factor, dtype=complex)
                for i in range(3):  # Sum over mass eigenstates
                    phase = np.exp(-1j * alpha_factor * self.dm2[i])
                    sum_terms += self.U[beta, i] * self.cU[alpha, i] * phase
                    d_sum_terms += (
                        self.d23_U[beta, i] * self.cU[alpha, i] * phase
                        + self.U[beta, i] * np.conj(self.d23_U[alpha, i]) * phase
                    )

                P[(alpha, beta)] = (
                    2 * np.abs(sum_terms) * np.abs(d_sum_terms)
                )  # Compute probability

        return P

    def dP_ddcp(self, L, E):
        alpha_factor = 1.267 * L / E  # Conversion factor for oscillation phase
        P = {}  # Store all transition probabilities

        for alpha in range(3):  # Initial flavors: 0=e, 1=μ, 2=τ
            for beta in range(3):  # Final flavors
                sum_terms = np.zeros_like(alpha_factor, dtype=complex)
                d_sum_terms = np.zeros_like(alpha_factor, dtype=complex)
                for i in range(3):  # Sum over mass eigenstates
                    phase = np.exp(-1j * alpha_factor * self.dm2[i])
                    sum_terms += self.U[beta, i] * self.cU[alpha, i] * phase
                    d_sum_terms += (
                        self.dd_U[beta, i] * self.cU[alpha, i] * phase
                        + self.U[beta, i] * np.conj(self.dd_U[alpha, i]) * phase
                    )

                P[(alpha, beta)] = (
                    2 * np.abs(sum_terms) * np.abs(d_sum_terms)
                )  # Compute probability

        return P


# Define the standard PMNS mixing angles (converted to radians)
theta_12 = np.radians(33.44)
theta_23 = np.radians(49.2)
theta_13 = np.radians(8.57)
delta_cp = np.radians(234)  # CP-violating phase

# Define mass-squared differences (in eV²)
Delta_m221 = 7.42e-5  # Solar mass-squared difference
Delta_m231 = 2.514e-3  # Atmospheric mass-squared difference
E = np.linspace(0.1, 1.0, 100)  # GeV
L = np.array([295.0])  # km

nu_mixing = Probability(theta_23, theta_13, theta_12, delta_cp, Delta_m221, Delta_m231)
nu_mixing.print_matrices()


# Define range of L and E
L_values = np.linspace(100, 1000, 50)  # Baselines from 100 km to 1000 km
E_values = np.linspace(0.1, 1.0, 50)  # Energy from 0.1 to 1.0 GeV
delta_cp_values = np.linspace(0, 2 * np.pi, 3)  # Test 3 delta_cp values

P_all = nu_mixing.value(L, E)

flavor_labels = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]

# Create and plot multiple 2D slices for different δ_CP values
fig, axs = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)

for k, delta_cp in enumerate(delta_cp_values):
    del nu_mixing
    nu_mixing = Probability(
        theta_23, theta_13, theta_12, delta_cp, Delta_m221, Delta_m231
    )
    P_matrix = nu_mixing.value(L_values, E_values)

    for α in range(3):
        for β in range(3):
            ax = axs[α, β]
            c = ax.imshow(
                P_matrix[:, :, α, β],
                aspect="auto",
                origin="lower",
                extent=[E_values.min(), E_values.max(), L_values.min(), L_values.max()],
                cmap="plasma",
                vmin=0,
                vmax=1,
            )

            if α == 2:
                ax.set_xlabel("Energy (GeV)")
            if β == 0:
                ax.set_ylabel("Baseline L (km)")

            ax.set_title(f"{flavor_labels[α]} → {flavor_labels[β]}")

fig.colorbar(c, ax=axs, orientation="vertical", label=r"$P_{\alpha \beta}$")
plt.suptitle(r"All Oscillation Probabilities $P_{\alpha \beta}$ vs. $L$ and $E$")
plt.show()
