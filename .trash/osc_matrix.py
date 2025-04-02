import sympy as sp
from sympy import preview

# Define the symbols for the angles and mass squared differences
theta_12, theta_23, theta_13, delta_CP, delta_m2_21, delta_m2_31, delta_m2_32, L, E = (
    sp.symbols(
        "theta_12 theta_23 theta_13 delta_CP delta_m2_21 delta_m2_31 delta_m2_32 L E"
    )
)

# Construct the PMNS matrix
U_PMNS = sp.Matrix(
    [
        [
            sp.cos(theta_13) * sp.cos(theta_12),
            sp.cos(theta_13) * sp.sin(theta_12),
            sp.sin(theta_13) * sp.exp(-sp.I * delta_CP),
        ],
        [
            -sp.sin(theta_13) * sp.cos(theta_23)
            - sp.cos(theta_13)
            * sp.sin(theta_23)
            * sp.sin(theta_12)
            * sp.exp(sp.I * delta_CP),
            sp.cos(theta_13) * sp.cos(theta_23)
            - sp.sin(theta_13)
            * sp.sin(theta_23)
            * sp.sin(theta_12)
            * sp.exp(sp.I * delta_CP),
            sp.sin(theta_23) * sp.sin(theta_12),
        ],
        [
            sp.sin(theta_13) * sp.sin(theta_23)
            - sp.cos(theta_13)
            * sp.cos(theta_23)
            * sp.sin(theta_12)
            * sp.exp(sp.I * delta_CP),
            -sp.sin(theta_13) * sp.cos(theta_23)
            - sp.cos(theta_13)
            * sp.sin(theta_23)
            * sp.sin(theta_12)
            * sp.exp(sp.I * delta_CP),
            sp.cos(theta_23) * sp.sin(theta_12),
        ],
    ]
)

# Create the mass difference matrix as a diagonal matrix with the mass differences
delta_m2_diag = sp.Matrix([[0, 0, 0], [0, delta_m2_21, 0], [0, 0, delta_m2_31]])

# Create the phase matrix for the mass differences, with L/E scaling
mass_phase = sp.diag(
    sp.exp(-sp.I * delta_m2_21 * L / (2 * E)),
    sp.exp(-sp.I * delta_m2_32 * L / (2 * E)),
    sp.exp(-sp.I * delta_m2_31 * L / (2 * E)),
)

# Multiply the PMNS matrix with the mass phase diagonal matrix
U_mixed = U_PMNS * mass_phase

# Compute the probability matrix P_alpha_to_beta
P_alpha_to_beta = (
    U_mixed * U_mixed.H
)  # Matrix multiplication of U_mixed with its conjugate transpose

# Simplify the resulting matrix (square the absolute values)
P_alpha_to_beta = sp.simplify(P_alpha_to_beta)

# Display the probability matrix
# sp.pprint(P_alpha_to_beta)
preview(P_alpha_to_beta)
