from gCHICOS import gCHICOS as gch

chic = gch()

# define hamiltonians with parameters
chic.set_vacuum_hamiltonian()
chic.set_matter_hamiltonian()
chic.set_NSI_hamiltonian()
chic.set_LV_hamiltonian()
chic.set_decoherence_hamiltonian()

# compute invariants and constant matrices
# only needs to be called after modifying hamiltonian parameters
chic.setup_constants()

# compute all needed quantities explcitly depending on energy and electron density
E = 0.1  # GeV
rho = 2.8  # g/cm³
Y_e = 0.5  # electron franction
chic.setup_physics(
    E, rho * Y_e
)  # there is no need to call it, it gets called when computing the oscillations

# compute oscillation probability matrix for fixed E, L and electron density
L = 295  # km
print("Osc. Prob. Matrix")
print(chic.compute_oscillations(E, L, rho * Y_e))


print("=======================================")
from CHICOS import CHICOS as ch
chic2 = ch()
print("Osc. Prob. Matrix")
print(chic2.oscillator([E], [L]))
