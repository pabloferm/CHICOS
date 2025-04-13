// gCHICOS.cpp
// Generalized CHICOS for any 3-flavor Hamiltonian
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <iostream>

using namespace Eigen;
using namespace std;

// Global constants
const double GF_factor = 7.56e-14;
const double SQRT3 = std::sqrt(3.0);

// Class definition
class gCHICOS {
public:
    // Constructor
    gCHICOS() {
        _construct_full_hamiltonian();
        need_update = true;
    }

    // Set the vacuum Hamiltonian
    void set_vacuum_hamiltonian(double theta_12 = 33.44,
                                double theta_23 = 49.2,
                                double theta_13 = 8.57,
                                double delta_cp = 234,
                                double dm2_21 = 7.42e-5,
                                double dm2_31 = 2.514e-3) {
        // Create the PMNS mixing matrix
        pmns_matrix(theta_12, theta_13, theta_23, delta_cp);

        // Construct the mass-squared difference matrix: 0.5 * diag(0, dm2_21, dm2_31)
        m2 = Matrix3d::Zero();
        m2(1, 1) = 0.5 * dm2_21;
        m2(2, 2) = 0.5 * dm2_31;

        // Compute Hvac = U * m2 * U†
        Matrix3cd Hvac = U * m2.cast<complex<double>>() * U.adjoint();
        H__10 += Hvac;
    }

    // Set the matter Hamiltonian
    void set_matter_hamiltonian() {
        // Create the matter potential matrix v = diag(GF_factor, 0, 0)
        Matrix3cd v = Matrix3cd::Zero();
        v(0, 0) = GF_factor;
        H_01 += v;
    }

    // Set the Non-Standard Interactions (NSI) Hamiltonian
    void set_NSI_hamiltonian(double eps_ee = 0.0,
                             double eps_emu = 0.0,
                             double eps_etau = 0.0,
                             double eps_mumu = 0.0,
                             double eps_mutau = 0.0,
                             double eps_tautau = 0.0) {
        Matrix3cd nsi;
        nsi(0, 0) = eps_ee;
        nsi(0, 1) = eps_emu;
        nsi(0, 2) = eps_etau;
        nsi(1, 0) = eps_emu;
        nsi(1, 1) = eps_mumu;
        nsi(1, 2) = eps_mutau;
        nsi(2, 0) = eps_etau;
        nsi(2, 1) = eps_mutau;
        nsi(2, 2) = eps_tautau;
        nsi *= GF_factor;
        H_01 += nsi;
    }

    // Set the Lorentz-violating (LV) Hamiltonian
    void set_LV_hamiltonian(double xi = 0.0) {
        Matrix3cd lv = Matrix3cd::Zero();
        lv(0, 1) = xi;
        lv(1, 0) = xi;
        H_10 += lv;
    }

    // Set the decoherence Hamiltonian
    void set_decoherence_hamiltonian(double gamma = 0.0) {
        Matrix3cd decoh = Matrix3cd::Zero();
        decoh(1, 1) = gamma;
        decoh(2, 2) = gamma;
        H_00 += decoh;
    }

    // Setup constant matrices and invariants.
    // (The invariants routine is left empty as in the original snippet.)
    void setup_constants() {
        _constant_matrices();
        _constant_invariants();
        need_update = false;
    }

    // Setup full physics with energy E and matter density rho.
    // (The invariants and matrices routines are left empty.)
    void setup_physics(double E, double rho) {
        if (need_update) {
            setup_constants();
        }
        _set_invariants(E, rho);
        _set_matrices(E, rho);
    }

    // For demonstration: print one of the Hamiltonian components.
    void print_H__10() const {
        cout << "H__10 matrix:" << endl << H__10 << endl;
    }

private:
    // Hamiltonian components (all 3x3 complex matrices)
    Matrix3cd H__10;  // Terms proportional to E⁻¹
    Matrix3cd H_00;   // Terms proportional to E⁰ and independent of the path/media
    Matrix3cd H_01;   // Terms proportional to E⁰ and dependent on the path/media
    Matrix3cd H_10;   // Terms proportional to E¹

    // Constant matrices for Hamiltonian squared
    Matrix3cd H2__20, H2__10, H2_00, H2_10, H2_20, H2__11, H2_01, H2_11, H2_02;

    // Constant matrices for Hamiltonian cubed
    Matrix3cd H3__30, H3__20, H3__10, H3_00, H3_10, H3_20, H3_30;
    Matrix3cd H3__21, H3__11, H3_01, H3_11, H3_21, H3__12, H3_02, H3_12, H3_03;

    // PMNS mixing matrix (3x3 complex) and the mass-squared diagonal matrix (3x3 real)
    Matrix3cd U;
    Matrix3d m2;

    // Flag to indicate whether constants need to be updated.
    bool need_update;

    // Private methods
    // Initializes the Hamiltonian components to zero.
    void _construct_full_hamiltonian() {
        H__10 = Matrix3cd::Zero();
        H_00  = Matrix3cd::Zero();
        H_01  = Matrix3cd::Zero();
        H_10  = Matrix3cd::Zero();
    }

    // Constructs the PMNS mixing matrix given the mixing angles (in degrees) and CP phase (also in degrees).
    void pmns_matrix(double theta_12, double theta_13, double theta_23, double delta_cp) {
        // Convert angles from degrees to radians.
        double t12 = theta_12 * M_PI / 180.0;
        double t13 = theta_13 * M_PI / 180.0;
        double t23 = theta_23 * M_PI / 180.0;
        double delta = delta_cp * M_PI / 180.0;

        // Calculate sines and cosines.
        double c12 = cos(t12);
        double s12 = sin(t12);
        double c13 = cos(t13);
        double s13 = sin(t13);
        double c23 = cos(t23);
        double s23 = sin(t23);

        // Define the complex phase.
        complex<double> exp_minus_i_delta = exp(complex<double>(0, -delta));
        complex<double> exp_plus_i_delta  = exp(complex<double>(0, delta));

        // Standard parameterization for the PMNS matrix.
        U = Matrix3cd::Zero();
        U(0, 0) = c12 * c13;
        U(0, 1) = s12 * c13;
        U(0, 2) = s13 * exp_minus_i_delta;

        U(1, 0) = -s12 * c23 - c12 * s23 * s13 * exp_plus_i_delta;
        U(1, 1) = c12 * c23 - s12 * s23 * s13 * exp_plus_i_delta;
        U(1, 2) = s23 * c13;

        U(2, 0) = s12 * s23 - c12 * c23 * s13 * exp_plus_i_delta;
        U(2, 1) = -c12 * s23 - s12 * c23 * s13 * exp_plus_i_delta;
        U(2, 2) = c23 * c13;
    }

    // Compute constant matrices for the Hamiltonian squared and cubed.
    void _constant_matrices() {
        // Hamiltonian squared terms
        H2__20 = H__10 * H__10;
        H2__10 = H__10 * H_00 + H_00 * H__10;
        H2_00  = H_00 * H_00 + H__10 * H_10 + H_10 * H__10;
        H2_10  = H_10 * H_00 + H_00 * H_10;
        H2_20  = H_10 * H_10;
        H2__11 = H__10 * H_01 + H_01 * H__10;
        H2_01  = H_00 * H_01 + H_01 * H_00;
        H2_11  = H_10 * H_01 + H_01 * H_10;
        H2_02  = H_01 * H_01;

        // Hamiltonian cubed terms needed for the explicit dependence of the determinant
        H3__30 = H__10 * H2__20;
        H3__20 = H__10 * H2__10 + H_00 * H2__20;
        H3__10 = H__10 * H2_00  + H_00 * H2__10 + H_10 * H2__20;
        H3_00  = H__10 * H2_10  + H_00 * H2_00  + H_10 * H2__10;
        H3_10  = H__10 * H2_20  + H_00 * H2_10  + H_10 * H2_00;
        H3_20  = H_00 * H2_20   + H_10 * H2_10;
        H3_30  = H_10 * H2_20;
        H3__21 = H__10 * H2__11 + H_01 * H2__20;
        H3__11 = H__10 * H2_01  + H_00 * H2__11 + H_01 * H2__10;
        H3_01  = H__10 * H2_11  + H_00 * H2_01  + H_01 * H2_00 + H_10 * H2__11;
        H3_11  = H_00 * H2_11   + H_01 * H2_10 + H_10 * H2_01;
        H3_21  = H_01 * H2_20   + H_10 * H2_11;
        H3__12 = H__10 * H2_02  + H_01 * H2__11;
        H3_02  = H_01 * H2_01   + H_00 * H2_02;
        H3_12  = H_01 * H2_11   + H_10 * H2_02;
        H3_03  = H_01 * H2_02;
    }

    // Empty stub for invariant constants (not provided in the original snippet)
    void _constant_invariants() {
        // Implement invariants if needed.
    }

    // Empty stub for setting invariants (depends on energy E and matter density rho)
    void _set_invariants(double E, double rho) {
        // Implement setting invariants as needed.
    }

    // Empty stub for setting matrices (depends on energy E and matter density rho)
    void _set_matrices(double E, double rho) {
        // Implement setting matrices as needed.
    }
};

// Main function for demonstration
int main() {
    // Create an instance of gCHICOS
    gCHICOS hamiltonian;

    // Set the vacuum Hamiltonian with default parameters
    hamiltonian.set_vacuum_hamiltonian();

    // Set matter effects
    hamiltonian.set_matter_hamiltonian();

    // Optionally, set additional physics effects:
    // For NSI effects:
    hamiltonian.set_NSI_hamiltonian(0.1, 0.05, 0.02, 0.0, 0.0, 0.0);
    // For Lorentz violation:
    hamiltonian.set_LV_hamiltonian(1e-3);
    // For decoherence:
    hamiltonian.set_decoherence_hamiltonian(1e-4);

    // Setup constants and physics with example energy and matter density
    hamiltonian.setup_physics(1.0, 3.0);
    
    // For demonstration, print one part of the Hamiltonian
    hamiltonian.print_H__10();
    
    return 0;
}
