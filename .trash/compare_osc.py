import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time
import nuSQuIDS as nsq

# Constants (example values, modify as needed)
Delta_m2_21 = 7.42e-5  # eV^2
Delta_m2_31 = 2.514e-3  # eV^2
sin2_theta_12 = 0.307
sin2_theta_13 = 0.0216
sin2_theta_23 = 0.545
L = 295  # Fixed baseline in km
E_vals = np.linspace(
    0.1, 5, 100
)  # Energy in GeV (avoid zero to prevent division issues)
V = 1.0e-13  # Matter potential (modify as needed)

# Compute mixing angles
sin_theta_12 = np.sqrt(sin2_theta_12)
sin_theta_13 = np.sqrt(sin2_theta_13)
sin_theta_23 = np.sqrt(sin2_theta_23)
cos_theta_12 = np.sqrt(1 - sin2_theta_12)
cos_theta_13 = np.sqrt(1 - sin2_theta_13)
cos_theta_23 = np.sqrt(1 - sin2_theta_23)

# Neutrino mixing matrix (PMNS)
U = np.array(
    [
        [cos_theta_12 * cos_theta_13, sin_theta_12 * cos_theta_13, sin_theta_13],
        [
            -sin_theta_12 * cos_theta_23 - cos_theta_12 * sin_theta_23 * sin_theta_13,
            cos_theta_12 * cos_theta_23 - sin_theta_12 * sin_theta_23 * sin_theta_13,
            sin_theta_23 * cos_theta_13,
        ],
        [
            sin_theta_12 * sin_theta_23 - cos_theta_12 * cos_theta_23 * sin_theta_13,
            -cos_theta_12 * sin_theta_23 - sin_theta_12 * cos_theta_23 * sin_theta_13,
            cos_theta_23 * cos_theta_13,
        ],
    ]
)


def exact_oscillation_probabilities(E):
    """Compute exact oscillation probabilities using matrix exponentiation."""
    M2 = np.diag([0, Delta_m2_21, Delta_m2_31])
    H_vac = M2 / (2 * E)
    H_mat = np.diag([V, 0, 0])
    H_full = U @ H_vac @ U.T + H_mat
    U_t = expm(-1j * H_full * L * 1.267)  # 1.267 = conversion factor for km/eV^2
    P = np.abs(U_t) ** 2  # Probability matrix
    return P


def approx_oscillation_probabilities(E):
    """Placeholder for approximate method"""
    return exact_oscillation_probabilities(E)  # Replace with actual approximation


def nusquids_oscillation_probabilities(E):
    """Compute oscillation probabilities using nuSQuIDS."""
    nuSQ = nsq.nuSQUIDS(3, nsq.NeutrinoType.neutrino)
    units = nsq.Const()
    nuSQ.Set_Body(nsq.Earth())
    nuSQ.Set_Track(nsq.Earth.Track(295.0 * units.km))
    nuSQ.Set_rel_error(1.0e-17)
    nuSQ.Set_abs_error(1.0e-17)
    nuSQ.Set_MixingParametersToDefault()
    nuSQ.Set_SquareMassDifference(2, 2.5e-3)
    nuSQ.Set_MixingAngle(1, 2, 0.6)
    nuSQ.Set_CPPhase(0, 2, 0.0)
    nuSQ.Set_Energy(E)
    nuSQ.EvolveState()
    P = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            P[i, j] = nusq.GetProb(i, j, False)
    return P


# Measure time for exact method
start_exact = time.time()
P_exact = np.array([exact_oscillation_probabilities(E) for E in E_vals])
end_exact = time.time()
time_exact = end_exact - start_exact

# Measure time for approximate method
start_approx = time.time()
P_approx = np.array([approx_oscillation_probabilities(E) for E in E_vals])
end_approx = time.time()
time_approx = end_approx - start_approx

# Measure time for nuSQuIDS method
start_nusquids = time.time()
P_nusquids = np.array([nusquids_oscillation_probabilities(E) for E in E_vals])
end_nusquids = time.time()
time_nusquids = end_nusquids - start_nusquids

# Compute differences
P_diff_exact_approx = P_exact - P_approx
P_diff_exact_nusquids = P_exact - P_nusquids

# Print computation times
print(f"Exact method computation time: {time_exact:.4f} seconds")
print(f"Approximate method computation time: {time_approx:.4f} seconds")
print(f"nuSQuIDS method computation time: {time_nusquids:.4f} seconds")

# Plot differences
flavors = ["e", "mu", "tau"]
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        ax.plot(
            E_vals, P_diff_exact_approx[:, i, j], label="Exact - Approx", linestyle="-"
        )
        ax.plot(
            E_vals,
            P_diff_exact_nusquids[:, i, j],
            label="Exact - nuSQuIDS",
            linestyle="--",
        )
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel(f"ΔP({flavors[i]} -> {flavors[j]})")
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.legend()
plt.tight_layout()
plt.show()
