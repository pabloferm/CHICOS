import sympy as sp
from sympy import preview, print_latex

# Define symbols
E, t = sp.symbols("E t")  # Energy and time
L = sp.Symbol("L")  # Distance traveled by neutrino

# Define mixing angles and mass differences
theta_12, theta_13, theta_23 = sp.symbols("theta_12 theta_13 theta_23")
Delta_m21, Delta_m31 = sp.symbols("Delta_m21 Delta_m31")

# PMNS mixing matrix
U = sp.Matrix(
    [
        [
            sp.cos(theta_12) * sp.cos(theta_13),
            sp.sin(theta_12) * sp.cos(theta_13),
            sp.sin(theta_13),
        ],
        [
            -sp.sin(theta_12) * sp.cos(theta_23)
            - sp.cos(theta_12) * sp.sin(theta_13) * sp.sin(theta_23),
            sp.cos(theta_12) * sp.cos(theta_23)
            - sp.sin(theta_12) * sp.sin(theta_13) * sp.sin(theta_23),
            sp.cos(theta_13) * sp.sin(theta_23),
        ],
        [
            sp.sin(theta_12) * sp.sin(theta_23)
            - sp.cos(theta_12) * sp.sin(theta_13) * sp.cos(theta_23),
            -sp.cos(theta_12) * sp.sin(theta_23)
            - sp.sin(theta_12) * sp.sin(theta_13) * sp.cos(theta_23),
            sp.cos(theta_13) * sp.cos(theta_23),
        ],
    ]
)

# Define mass eigenstates evolution
M_diag = sp.diag(
    0, Delta_m21 / (2 * E), Delta_m31 / (2 * E)
)  # Mass-squared differences
U_dagger = U.H  # Hermitian conjugate

# Compute the full oscillation probability matrix
S = U * sp.exp(-sp.I * M_diag * L) * U_dagger  # Time evolution matrix
P_matrix = sp.simplify(sp.Abs(S) ** 2)  # Probability matrix

print("Oscillation Probability Matrix:")
sp.pprint(P_matrix)
# print_latex(P_matrix)
