# Generalized CHICOS for any 3-flavor Hamiltonian

import numpy as np

GF_factor = 7.56e-14
SQRT3 = np.sqrt(3)

class gCHICOS:
    def __init__(self):
        self._construct_full_hamiltonian()
        self.need_update = True

    def set_vacuum_hamiltonian(
        self,
        theta_12=33.44,
        theta_23=49.2,
        theta_13=8.57,
        delta_cp=234,
        dm2_21=7.42e-5,
        dm2_31=2.514e-3,
    ):
        self.pmns_matrix(theta_12=theta_12, theta_13=theta_13, theta_23=theta_23, delta_cp=delta_cp)
        self.m2 = 0.5 * np.diag([0, dm2_21, dm2_31])
        Hvac = self.U @ self.m2 @ np.conj(self.U).T
        self.H__10 = self.H__10 + Hvac

    def set_matter_hamiltonian(self):
        self.v = np.diag([GF_factor, 0, 0])
        self.H_01 = self.H_01 + self.v

    def set_NSI_hamiltonian(
        self,
        eps_ee=0.0,
        eps_emu=0.0,
        eps_etau=0.0,
        eps_mumu=0.0,
        eps_mutau=0.0,
        eps_tautau=0.0,
    ):
        self.nsi = GF_factor * np.array([
                [eps_ee, eps_emu, eps_etau],
                [eps_emu, eps_mumu, eps_mutau],
                [eps_etau, eps_mutau, eps_tautau],
            ])
        self.H_01 = self.H_01 + self.nsi

    def set_LV_hamiltonian(self, xi=0.0):
        self.lv = xi * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        self.H_10 = self.H_10 + self.lv

    def set_decoherence_hamiltonian(self, gamma=0.0):
        self.decoh = np.diag([0, gamma, gamma])
        self.H_00 = self.H_00 + self.decoh

    def _construct_full_hamiltonian(self):
        self.H__10 = np.zeros((3, 3))  # Hamiltonian terms proportional to E⁻¹
        self.H_00 = np.zeros((3, 3))  # Hamiltonian terms proportional to E⁰ and not depend on the path/media
        self.H_01 = np.zeros((3, 3))  # Hamiltonian terms proportional to E⁰ and depend on the path/media
        self.H_10 = np.zeros((3, 3))  # Hamiltonian terms proportional to E¹

    def setup_constants(self):
        self._constant_matrices()
        self._constant_invariants()
        self.need_update = False

    def setup_physics(self, E, rho):
        if self.need_update:
            self.setup_constants()
        self._set_invariants(E, rho)
        self._set_matrices(E, rho)

    def _constant_matrices(self):
        # Terms for Hamiltonian squared
        self.H2__20 = self.H__10 @ self.H__10
        self.H2__10 = self.H__10 @ self.H_00 + self.H_00 @ self.H__10
        self.H2_00 = self.H_00 @ self.H_00 + self.H__10 @ self.H_10 + self.H_10 @ self.H__10
        self.H2_10 = self.H_10 @ self.H_00 + self.H_00 @ self.H_10
        self.H2_20 = self.H_10 @ self.H_10
        self.H2__11 = self.H__10 @ self.H_01 + self.H_01 @ self.H__10
        self.H2_01 = self.H_00 @ self.H_01 + self.H_01 @ self.H_00
        self.H2_11 = self.H_10 @ self.H_01 + self.H_01 @ self.H_10
        self.H2_02 = self.H_01 @ self.H_01

        # Terms for Hamiltonian cubed, needed for the explicit dependence of the determinant
        self.H3__30 = self.H__10 @ self.H2__20
        self.H3__20 = self.H__10 @ self.H2__10 + self.H_00 @ self.H2__20
        self.H3__10 = self.H__10 @ self.H2_00  + self.H_00 @ self.H2__10 + self.H_10 @ self.H2__20
        self.H3_00  = self.H__10 @ self.H2_10  + self.H_00 @ self.H2_00  + self.H_10 @ self.H2__10
        self.H3_10  = self.H__10 @ self.H2_20  + self.H_00 @ self.H2_10  + self.H_10 @ self.H2_00
        self.H3_20  = self.H_00 @ self.H2_20   + self.H_10 @ self.H2_10
        self.H3_30  = self.H_10 @ self.H2_20
        self.H3__21 = self.H__10 @ self.H2__11 + self.H_01 @ self.H2__20
        self.H3__11 = self.H__10 @ self.H2_01  + self.H_00 @ self.H2__11 + self.H_01 @ self.H2__10
        self.H3_01  = self.H__10 @ self.H2_11  + self.H_00 @ self.H2_01  + self.H_01 @ self.H2_00 + self.H_10 @ self.H2__11
        self.H3_11  = self.H_00 @ self.H2_11   + self.H_01 @ self.H2_10 + self.H_10 @ self.H2_01
        self.H3_21  = self.H_01 @ self.H2_20   + self.H_10 @ self.H2_11
        self.H3__12 = self.H__10 @ self.H2_02  + self.H_01 @ self.H2__11
        self.H3_02  = self.H_01 @ self.H2_01   + self.H_00 @ self.H2_02
        self.H3_12  = self.H_01 @ self.H2_11 + self.H_10 @ self.H2_02
        self.H3_03  = self.H_01 @ self.H2_02

    def _set_matrices(self, E, rho):
        # Hamiltonian
        self.H = self.H__10 / E + self.H_00 + self.H_01 * rho + self.H_10 * E
        self.H2 = self.H2__20 / E**2 + self.H2__10 / E + self.H2_00
        + self.H2_10 * E + self.H2_20 * E**2 + self.H2__11 * rho / E
        + self.H2_01 * rho + self.H2_11 * rho * E
        self.H3 = self.H3__30 / E**3 + self.H3__20 / E**2 + self.H3__10 / E + self.H3_00
        + self.H3_10 * E + self.H3_20 * E**2 + self.H3_30 * E**3 + self.H3__21 * rho / E**2
        + self.H3__11 * rho / E + self.H3_01 * rho + self.H3_11 * rho * E + self.H3_21 * rho * E**2
        + self.H3__12 * rho**2 / E + self.H3_02 * rho**2 + self.H3_12 * rho**2 * E
        + self.H3_03 * rho**3

        # Shifted Hamiltonian
        self.Hs = self.H - self.TrH / 3 * np.identity(3)
        self.Hs2 = self.H2 - 2/3 * self.TrH * self.H + (self.TrH / 3)**2 * np.identity(3)

    def _constant_invariants(self):
        # Traces of Hamiltonian
        self.TrH__10 = np.trace(self.H__10)
        self.TrH_00 = np.trace(self.H_00)
        self.TrH_01 = np.trace(self.H_01)
        self.TrH_10 = np.trace(self.H_10)

        # Traces of Hamiltonian squared
        self.TrH2__20 = np.trace(self.H2__20)
        self.TrH2__10 = np.trace(self.H2__10)
        self.TrH2_00 = np.trace(self.H2_00)
        self.TrH2_10 = np.trace(self.H2_10)
        self.TrH2_20 = np.trace(self.H2_20)
        self.TrH2__11 = np.trace(self.H2__11)
        self.TrH2_01 = np.trace(self.H2_01)
        self.TrH2_11 = np.trace(self.H2_11)
        self.TrH2_02 = np.trace(self.H2_02)

        # Traces of Hamiltonian cubed
        self.TrH3__30 = np.trace(self.H3__30)
        self.TrH3__20 = np.trace(self.H3__20)
        self.TrH3__10 = np.trace(self.H3__10)
        self.TrH3_00  = np.trace(self.H3_00)
        self.TrH3_10  = np.trace(self.H3_10)
        self.TrH3_20  = np.trace(self.H3_20)
        self.TrH3_30  = np.trace(self.H3_30)
        self.TrH3__21 = np.trace(self.H3__21)
        self.TrH3__11 = np.trace(self.H3__11)
        self.TrH3_01  = np.trace(self.H3_01)
        self.TrH3_11  = np.trace(self.H3_11)
        self.TrH3_21  = np.trace(self.H3_21)
        self.TrH3__12 = np.trace(self.H3__12)
        self.TrH3_02  = np.trace(self.H3_02)
        self.TrH3_12  = np.trace(self.H3_12)
        self.TrH3_03  = np.trace(self.H3_03)

    def _set_invariants(self, E, rho):
        # Invariants of Hamiltonian
        self.TrH = np.real(self.TrH__10 / E + self.TrH_00 + self.TrH_01 * rho + self.TrH_10 * E)
        self.TrH2 = np.real(self.TrH2__20 / E**2 + self.TrH2__10 / E + self.TrH2_00 + self.TrH2_10 * E)
        + self.TrH2_20 * E**2 + self.TrH2__11 * rho / E + self.TrH2_01 * rho 
        + self.TrH2_11 * rho * E + self.TrH2_02 * rho**2
        TrH3 = self.TrH3__30 / E**3 + self.TrH3__20 / E**2 + self.TrH3__10 / E + self.TrH3_00
        + self.TrH3_10 * E + self.TrH3_20 * E**2 + self.TrH3_30 * E**3 + self.TrH3__21 * rho / E**2
        + self.TrH3__11 * rho / E + self.TrH3_01 * rho + self.TrH3_11 * rho * E + self.TrH3_21 * rho * E**2
        + self.TrH3__12 * rho**2 / E + self.TrH3_02 * rho**2 + self.TrH3_12 * rho**2 * E
        + self.TrH3_03 * rho**3
        self.DetH = np.real(TrH3 / 3 - 0.5 * self.TrH * self.TrH2 + (self.TrH**3) / 6)

        # Invariants of shifted Hamiltonian
        self.TrHs2 = self.TrH2 - (self.TrH**2) / 3
        self.DetHs = self.DetH + self.TrH2 * self.TrH / 6 - 5 / 54 * self.TrH**3
        self.Disc2_Hs = np.real(0.5 * self.TrHs2**3 - 27 * self.DetHs**2) # why real part needed!?

        # Eigenvalues of shifted Hamiltonian
        _theta = np.arccos(np.sqrt(54 * self.DetHs**2 / self.TrHs2**3))
        _sin_theta = np.sin(_theta/3)
        _cos_theta = np.sqrt(1 - _sin_theta**2)
        _scale = np.sqrt(2 * self.TrHs2 / 3)
        self.lambdas = np.zeros(3)
        self.lambdas[0] = _scale * _cos_theta
        self.lambdas[1] = -0.5 * _scale * (_cos_theta + SQRT3 * _sin_theta)
        self.lambdas[2] = -self.lambdas[0] - self.lambdas[1]

    def compute_oscillations(self, E, L, rho, alpha=None, beta=None):  
        # optional arguments for specific channel(s)
        S = self._amplitude(E, L, rho)
        return np.abs(S) ** 2 / self.Disc2_Hs

    def _amplitude(self, E, L, rho):
        self.setup_physics(E, rho)
        L *= 1.267 # convert to proper units
        J = 0
        for i in range(3):
            diff_exp = (self.lambdas[i - 1] - self.lambdas[i - 2]) * np.exp(-1j * L * self.lambdas[i])
            l_jk = self.lambdas[i - 1] * self.lambdas[i - 2]
            l_i = self.lambdas[i]
            J += diff_exp * ((l_jk * np.identity(3) + l_i * self.Hs) + self.Hs2)
        return J

    def pmns_matrix(self, theta_12=0, theta_13=0, theta_23=0, delta_cp=0):
        theta_12 = np.radians(theta_12)
        theta_23 = np.radians(theta_23)
        theta_13 = np.radians(theta_13)
        delta_cp = np.radians(delta_cp)
        c12, s12 = np.cos(theta_12), np.sin(theta_12)
        c23, s23 = np.cos(theta_23), np.sin(theta_23)
        c13, s13 = np.cos(theta_13), np.sin(theta_13)
        e_idelta = np.exp(1j * delta_cp)

        self.U = np.array([
                [c12 * c13, s12 * c13, s13 * np.conj(e_idelta)],
                [-s12 * c23 - c12 * s23 * s13 * e_idelta, c12 * c23 - s12 * s23 * s13 * e_idelta, s23 * c13,],
                [s12 * s23 - c12 * c23 * s13 * e_idelta, -c12 * s23 - s12 * c23 * s13 * e_idelta, c23 * c13,],
            ])