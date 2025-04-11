import numpy as np
from gCHICOS import gCHICOS
from bodies import Earth


class AtmgCHICOS(gCHICOS):
    """docstring for AtmCHICOS"""

    def __init__(self, earth_profile="7p5percent"):
        super(AtmgCHICOS, self).__init__()
        self.earth = Earth(label=earth_profile)

    def _total_amplitude(self, E, cos_zentih):
        segs, rhos = self.earth.path(cos_zentih)
        S = np.diag([1.0, 1.0, 1.0])
        D = 1.0
        for s, r in zip(segs, rhos):
            S = np.matmul(S, self._amplitude(E, s, r))
            D *= self.Disc2_Hs
        return S, D

    def compute_oscillations(
        self, E, cos_zentih, alpha=None, beta=None
    ):  # optional arguments for specific channel(s)
        J, D = self._total_amplitude(E, cos_zentih)
        if D == 0:
            D = 5e-324
        return np.abs(J) ** 2 / D

    def oscillator(self, E_array, CZ_array):
        return np.array(
            [self.compute_oscillations(E, CZ) for (E, CZ) in zip(E_array, CZ_array)]
        )
