import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time
from CHICOS import CHICOS as ch
from CHICOS_numpy import npCHICOS as npch


# Constants (example values, modify as needed)
Delta_m2_21 = 7.42e-5  # eV^2
Delta_m2_31 = 2.514e-3  # eV^2
theta_12=np.radians(33.44)
theta_23=np.radians(49.2)
theta_13=np.radians(8.57)
delta_cp=np.radians(234)
cos_theta_12, sin_theta_12 = np.cos(theta_12), np.sin(theta_12)
cos_theta_23, sin_theta_23 = np.cos(theta_23), np.sin(theta_23)
cos_theta_13, sin_theta_13 = np.cos(theta_13), np.sin(theta_13)
L = 295  # Fixed baseline in km
E_vals = np.geomspace(0.01, 10, 10000) # Energy in GeV (avoid zero to prevent division issues)
G_F = 1.1663787e-5  # Fermi constant in eV⁻²
Y_e = 0.5  # Electron fraction
V = 7.56e-14 * Y_e * 2.8

# Neutrino mixing matrix (PMNS)
U = np.array(
    [
        [cos_theta_12 * cos_theta_13, sin_theta_12 * cos_theta_13, sin_theta_13*np.exp(-1j * delta_cp)],
        [
            -sin_theta_12 * cos_theta_23 - cos_theta_12 * sin_theta_23 * sin_theta_13 * np.exp(1j * delta_cp),
            cos_theta_12 * cos_theta_23 - sin_theta_12 * sin_theta_23 * sin_theta_13 * np.exp(1j * delta_cp),
            sin_theta_23 * cos_theta_13,
        ],
        [
            sin_theta_12 * sin_theta_23 - cos_theta_12 * cos_theta_23 * sin_theta_13 * np.exp(1j * delta_cp),
            -cos_theta_12 * sin_theta_23 - sin_theta_12 * cos_theta_23 * sin_theta_13 * np.exp(1j * delta_cp),
            cos_theta_23 * cos_theta_13,
        ],
    ]
)

def exact_oscillation_probabilities(E):
    """Compute exact oscillation probabilities using matrix exponentiation."""
    M2 = np.diag([0, Delta_m2_21, Delta_m2_31])
    H_vac = M2 / (2 * E)
    H_mat = np.diag([V, 0, 0])
    H_full = U @ H_vac @ np.conj(U).T + H_mat
    U_t = expm(-1j * H_full * L * 1.267)  # 1.267 = conversion factor for km/eV^2
    return np.abs(U_t) ** 2  # Probability matrix

def approx_oscillation_probabilities(E):
    """Placeholder for approximate method"""
    baselines = np.zeros_like(E_vals) + 295  # km
    #chic = ch()
    chic = npch()
    return chic.oscillator(E_vals, baselines)


# Measure time for exact method
start_exact = time.time()
P_exact = np.array([exact_oscillation_probabilities(E) for E in E_vals])
end_exact = time.time()
time_exact = end_exact - start_exact

# Measure time for approximate method
start_approx = time.time()
P_approx = approx_oscillation_probabilities(E_vals)
end_approx = time.time()
time_approx = end_approx - start_approx

# Compute differences
P_diff = np.abs(P_exact - P_approx)

# Print computation times
print(f"Exact method computation time: {time_exact:.4f} seconds")
print(f"Approximate method computation time: {time_approx:.4f} seconds")
print(f"That is {time_exact/time_approx:.4f} faster")

# Plot differences
flavors = ["e", "mu", "tau"]
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        #ax.plot(E_vals, P_exact[:, i, j], label="Exact", linestyle="-")
        #ax.plot(E_vals, P_approx[:, i, j], label="CHICOS", linestyle="-")
        ax.plot(E_vals, P_diff[:, i, j], label="CHICOS-pade", linestyle="-")
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel(f"ΔP({flavors[i]} -> {flavors[j]})")
        #ax.set_yscale("log")
        ax.set_xscale("log")
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.legend()
plt.tight_layout()
plt.show()
