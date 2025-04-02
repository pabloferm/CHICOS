import numpy as np
import matplotlib.pyplot as plt


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

        self.M = np.diag([0, Delta_m221, Delta_m231])
        self.dm2 = [Delta_m221, Delta_m231, Delta_m231 - Delta_m221]

    def value(self, L, E):
        alpha_factor = 1.267 * L / E  # Conversion factor for oscillation phase
        P = {}  # Store all transition probabilities

        for alpha in range(3):  # Initial flavors: 0=e, 1=μ, 2=τ
            for beta in range(3):  # Final flavors
                sum_terms = np.zeros_like(alpha_factor, dtype=complex)
                for i in range(3):  # Sum over mass eigenstates
                    phase = np.exp(-1j * alpha_factor * self.dm2[i])
                    sum_terms += self.U[beta, i] * self.cU[alpha, i] * phase

                P[(alpha, beta)] = np.abs(sum_terms) ** 2  # Compute probability

        return P

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


if __name__ == "__main__":
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

    nu_mixing = Probability(
        theta_23, theta_13, theta_12, delta_cp, Delta_m221, Delta_m231
    )
    nu_mixing.print_matrices()

    P_all = nu_mixing.value(L, E)
    d23P_all = nu_mixing.dP_ddcp(L, E)

    # Define flavor labels
    flavors = {0: "e", 1: "μ", 2: "τ"}

    # Plot all oscillation probabilities
    fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=True)

    for (alpha, beta), P in P_all.items():
        axs[0].plot(E, P, label=rf"$P(\nu_{flavors[alpha]} \to \nu_{flavors[beta]})$")
        axs[1].plot(
            E,
            d23P_all[(alpha, beta)],
            label=rf"$dP(\nu_{flavors[alpha]} \to \nu_{flavors[beta]})$",
        )
        # axs[2].plot(E, d23P_all[(alpha, beta)]/P, label=rf"$dP(\nu_{flavors[alpha]} \to \nu_{flavors[beta]})$")

    axs[1].set_xlabel("Energy (GeV)")
    axs[0].set_ylabel("Oscillation Probability")
    axs[1].set_ylabel("Diff. Oscillation Probability")
    # plt.title("Three-Flavor Neutrino Oscillation Probabilities in Vacuum")
    axs[0].legend()
    # plt.grid()
    plt.show()

    delta_cp_values = np.linspace(0, 2 * np.pi, 100)
    theta_23_values = np.linspace(np.radians(35), np.radians(75), 100)
    E_values = np.linspace(0.1, 1.0, 100)  # Energy range (GeV)

    P_matrix = np.zeros((len(delta_cp_values), len(E_values)))

    # for i, delta_cp in enumerate(delta_cp_values):
    for i, theta_23 in enumerate(theta_23_values):
        del nu_mixing
        nu_mixing = Probability(
            theta_23, theta_13, theta_12, delta_cp, Delta_m221, Delta_m231
        )
        for j, E in enumerate(E_values):
            # delP = nu_mixing.value(L, E)
            # delP = nu_mixing.dP_ddcp(L, E)
            delP = nu_mixing.dP_d23(L, E)
            P_matrix[i, j] = delP[1, 1] + delP[1, 0]

    # Create 2D density plot
    plt.figure(figsize=(10, 6))
    # plt.imshow(P_matrix, aspect='auto', origin='lower', extent=[E_values.min(), E_values.max(), np.sin(theta_23_values.min())**2, np.sin(theta_23_values.max())**2], cmap='viridis')
    (
        X,
        Y,
    ) = np.meshgrid(E_values, np.sin(theta_23_values) ** 2)
    plt.contour(
        X,
        Y,
        P_matrix,
        aspect="auto",
        origin="lower",
        extent=[
            E_values.min(),
            E_values.max(),
            np.sin(theta_23_values.min()) ** 2,
            np.sin(theta_23_values.max()) ** 2,
        ],
        cmap="viridis",
    )

    # Labels and colorbar
    plt.colorbar(label=r"$P(\nu_\mu \to \nu_e)$")
    plt.xlabel("Neutrino Energy (GeV)")
    # plt.ylabel(r"CP Phase $\delta_{CP}$ (radians)")
    plt.ylabel(r"$\theta_{23}$ (radians)")
    # plt.title(r"Oscillation Probability $P(\nu_\mu \to \nu_e)$ vs. Energy and $\delta_{CP}$")

    # Adjust y-ticks to show π fractions
    # plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    # [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    plt.show()
