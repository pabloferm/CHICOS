import numpy as np
from CHICOS import CHICOS
from earth import Earth


class AtmCHICOS(CHICOS):
    """docstring for AtmCHICOS"""

    def __init__(self, earth_profile="7p5percent"):
        super(AtmCHICOS, self).__init__()
        self.earth = Earth(label=earth_profile)

    def _total_amplitude(self, E, cos_zentih):
        segs, rhos = self.earth.path(cos_zentih)
        S = np.diag([1.0, 1.0, 1.0])
        for s, r in zip(segs, rhos):
            self.set_density(r, 1)
            S = np.matmul(S, self._amplitude(E, s)) / np.sqrt(self.Disc2_Hs)
        return S

    def compute_oscillations(self, E, cos_zentih, alpha=None, beta=None):
    	# optional arguments for specific channel(s)
        S = self._total_amplitude(E, cos_zentih)
        return np.abs(S) ** 2

    def oscillator(self, E_array, CZ_array):
        return np.array(
            [self.compute_oscillations(E, CZ) for (E, CZ) in zip(E_array, CZ_array)]
        )
