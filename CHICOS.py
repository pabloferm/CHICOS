import numpy as np


class CHICOS:
    def __init__(
        self,
        theta_12=33.44,
        theta_23=49.2,
        theta_13=8.57,
        delta_cp=234,
        dm2_21=7.42e-5,
        dm2_31=2.514e-3,
        density=2.8,  # g/cm³
    ):
        # Matter effects
        G_F = 1.1663787e-5  # Fermi constant in eV⁻²
        Y_e = 0.5  # Electron fraction
        self.V = 7.56e-14 * Y_e * density # Convert to eV

        self.theta_12 = np.radians(theta_12)
        self.theta_23 = np.radians(theta_23)
        self.theta_13 = np.radians(theta_13)
        self.delta_cp = np.radians(delta_cp)
        self.dm2_21 = dm2_21
        self.dm2_31 = dm2_31
        self.need_update = True
        self._set_matrices()

    def oscillator(self, E_array, L_array):
        return np.array([self.compute_oscillations(E, L) for (E, L) in zip(E_array, L_array)])

    def compute_oscillations(self, E, L, alpha=None, beta=None):  # optional arguments for specific channel(s)
        J = self._amplitude(E, L)
        return np.abs(J) ** 2 if self.Disc2_Hs == 0 else np.abs(J) ** 2 / self.Disc2_Hs

    # TODO Rename this here and in `compute_oscillations` and `test`
    def _amplitude(self, E, L):
        self._set_invariants(E)
        self.shift_hamiltonian(E)
        self.shift_hamiltonian_squared(E)
        L *= 1.267
        result = 0
        for i in range(3):
            diff_exp = (self.lambdas[i - 1] - self.lambdas[i - 2]) * np.exp(-1j * L * self.lambdas[i])
            l_jk = self.lambdas[i - 1] * self.lambdas[i - 2]
            l_i = self.lambdas[i]
            result += diff_exp * (((l_jk * np.identity(3) + l_i * self.Hs) + self.Hs2))
        return result

    def shift_hamiltonian(self, E): # checked
        self.Hs = self.hamiltonian(E) - self.TrH * np.identity(3) / 3

    def shift_hamiltonian_squared(self, E): # checked
        self.Hs2 = self.hamiltonian_squared(E) - 2 * self.TrH * self.hamiltonian(E) / 3 + self.TrH**2 * np.identity(3) / 9

    def hamiltonian(self, E): # checked
        if self.need_update:
            self._set_matrices()
        return self.um2u / (2 * E) + self.v

    def hamiltonian_squared(self, E): # checked
        if self.need_update:
            self._set_matrices()
        return self.um4u / (4 * E**2) + self.v2 + (self.um2uv + self.vum2u) / (2* E)

    def _set_invariants(self, E):
        self.TrH = (self.dm2_21 + self.dm2_31) / (2 * E) + self.V
        self.TrH2 = (self.dm2_21**2 + self.dm2_31**2) / (4 * E**2) + self.V**2 + self.V * (np.abs(self.U[0, 1])**2 * self.dm2_21 + np.abs(self.U[1, 2])**2 * self.dm2_31) / E
        self.DetH = self.V * self.dm2_21 * self.dm2_31 * np.abs(self.U[1, 1] * self.U[2, 2] - self.U[1, 2] * self.U[2, 1])**2 / (2 * E)**2
        self.TrHs2 = self.TrH2 - self.TrH**2 / 3
        self.DetHs = self.DetH + self.TrH2 * self.TrH / 6 - 5 / 54 * self.TrH**3
        self.Disc2_Hs = 0.5 * self.TrHs2 ** 3 - 27 * self.DetHs ** 2
        
        _theta = np.arccos(np.sqrt(54 * self.DetHs**2 / self.TrHs2**3))
        _scale = np.sqrt(2 * self.TrHs2 / 3)
        self.lambdas = np.array([_scale * np.cos((_theta + 2 * np.pi * k) / 3) for k in range(3)])

    def _set_matrices(self):
        self.pmns_matrix()
        self.m2 = np.diag([0, self.dm2_21, self.dm2_31])
        self.um2u = self.U @ self.m2 @ np.conj(self.U).T
        self.um4u = self.U @ np.diag([0, self.dm2_21**2, self.dm2_31**2]) @ np.conj(self.U).T
        self.v2 = np.diag([self.V**2, 0, 0])
        self.v = np.diag([self.V, 0, 0])
        self.um2uv = self.um2u @ self.v
        self.vum2u = self.v @ self.um2u
        self.need_update = False

    def pmns_matrix(self):
        c12, s12 = np.cos(self.theta_12), np.sin(self.theta_12)
        c23, s23 = np.cos(self.theta_23), np.sin(self.theta_23)
        c13, s13 = np.cos(self.theta_13), np.sin(self.theta_13)
        e_idelta = np.exp(1j * self.delta_cp)

        self.U = np.array([
                [c12 * c13, s12 * c13, s13 * np.conj(e_idelta)],
                [-s12 * c23 - c12 * s23 * s13 * e_idelta, c12 * c23 - s12 * s23 * s13 * e_idelta, s23 * c13,],
                [s12 * s23 - c12 * c23 * s13 * e_idelta, -c12 * s23 - s12 * c23 * s13 * e_idelta, c23 * c13,],
            ])
