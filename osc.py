import sympy as sp
from sympy import preview

# Define the symbols for the angles and mass squared differences
theta_12, theta_23, theta_13, delta_CP, delta_m2_21, delta_m2_31, delta_m2_32, L, E = sp.symbols(
    'theta_12 theta_23 theta_13 delta_CP delta_m2_21 delta_m2_31 delta_m2_32 L E')

# Construct the PMNS matrix
U_PMNS = sp.Matrix([
    [sp.cos(theta_13) * sp.cos(theta_12), sp.cos(theta_13) * sp.sin(theta_12), sp.sin(theta_13) * sp.exp(-sp.I * delta_CP)],
    [-sp.sin(theta_13) * sp.cos(theta_23) - sp.cos(theta_13) * sp.sin(theta_23) * sp.sin(theta_12) * sp.exp(sp.I * delta_CP), 
     sp.cos(theta_13) * sp.cos(theta_23) - sp.sin(theta_13) * sp.sin(theta_23) * sp.sin(theta_12) * sp.exp(sp.I * delta_CP), 
     sp.sin(theta_23) * sp.sin(theta_12)],
    [sp.sin(theta_13) * sp.sin(theta_23) - sp.cos(theta_13) * sp.cos(theta_23) * sp.sin(theta_12) * sp.exp(sp.I * delta_CP), 
     -sp.sin(theta_13) * sp.cos(theta_23) - sp.cos(theta_13) * sp.sin(theta_23) * sp.sin(theta_12) * sp.exp(sp.I * delta_CP),
     sp.cos(theta_23) * sp.sin(theta_12)]
])

# Calculate the oscillation probabilities P_alpha_to_beta
P_alpha_to_beta = sp.zeros(3, 3)

# Calculate each probability P_alpha_to_beta
for alpha in range(3):
    for beta in range(3):
        sum_terms = 0
        for i in range(3):
            # The mass eigenstate difference terms (diagonal mass differences)
            mass_term = sp.exp(-sp.I * (delta_m2_31 if i == 0 else delta_m2_32 if i == 1 else delta_m2_21) * L / (2 * E))
            sum_terms += U_PMNS[alpha, i] * sp.conjugate(U_PMNS[beta, i]) * mass_term
        P_alpha_to_beta[alpha, beta] = sp.simplify(sp.Abs(sum_terms)**2)

# Display the probability matrix
preview(P_alpha_to_beta)

