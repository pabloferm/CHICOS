#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <iomanip>

using namespace Eigen;
using namespace std;

const double PI = acos(-1);

class CHICOS {
public:
    // Constructor with default parameters (angles in degrees)
    CHICOS(double theta_12_deg = 33.44,
           double theta_23_deg = 49.2,
           double theta_13_deg = 8.57,
           double delta_cp_deg = 234,
           double dm2_21 = 7.42e-5,
           double dm2_31 = 2.514e-3,
           double density = 2.8,   // g/cm^3
           double Y_e = 0.5)       // unitless (effective electron fraction)
    {
        theta_12 = theta_12_deg * PI / 180.0;
        theta_23 = theta_23_deg * PI / 180.0;
        theta_13 = theta_13_deg * PI / 180.0;
        delta_cp = delta_cp_deg * PI / 180.0;
        this->dm2_21 = dm2_21;
        this->dm2_31 = dm2_31;
        
        // Matter effects
        set_density(density, Y_e);
        
        need_update = true;
        _set_matrices();
    }
    
    // Set the density / matter effects.
    void set_density(double density, double Y_e) {
        // Fermi constant G_F = 1.1663787e-5 (eV^-2) not used explicitly here.
        // The matter potential V is given by: 7.56e-14 * Y_e * density (in eV).
        V = 7.56e-14 * Y_e * density;
    }
    
    // Update matrices (calls _set_matrices)
    void update() {
        _set_matrices();
    }
    
    // Oscillator: takes arrays of energy and baseline, returns a vector of (3x3) oscillation matrices.
    vector<Matrix3d> oscillator(const vector<double>& E_array, const vector<double>& L_array) {
        size_t n = E_array.size();
        vector<Matrix3d> results;
        results.reserve(n);
        for (size_t i = 0; i < n; i++) {
            results.push_back(compute_oscillations(E_array[i], L_array[i]));
        }
        return results;
    }
    
    // Compute oscillations for a given energy E and baseline L.
    // Optional channel parameters (alpha, beta) are not used in this implementation.
    Matrix3d compute_oscillations(double E, double L, int alpha = -1, int beta = -1) {
        Matrix3cd J = _amplitude(E, L);
        // Element-wise square of the absolute value.
        Matrix3d prob = Matrix3d::Zero();
        for (int i = 0; i < J.rows(); i++) {
            for (int j = 0; j < J.cols(); j++) {
                double mod2 = norm(J(i, j)); // squared modulus
                prob(i, j) = mod2 / Disc2_Hs;
            }
        }
        return prob;
    }
    
private:
    // Mixing angles and CP phase (in radians)
    double theta_12, theta_23, theta_13, delta_cp;
    // Mass squared differences and matter potential
    double dm2_21, dm2_31, V;
    
    // Invariants
    double TrH, TrH2, DetH, TrHs2, DetHs, Disc2_Hs;
    
    // Eigenvalues (lambdas)
    Vector3d lambdas;
    
    // Matrices (all 3x3 matrices)
    Matrix3cd U;           // PMNS matrix
    Matrix3cd Hs;          // Shifted Hamiltonian
    Matrix3cd Hs2;         // Shifted squared Hamiltonian
    Matrix3cd v;           // Matter potential matrix (diagonal)
    Matrix3cd v2;          // Square of matter potential matrix
    Matrix3cd um2u;        // U * m2 * U^dagger, with m2 = diag(0, dm2_21, dm2_31)
    Matrix3cd um4u;        // U * diag(0, dm2_21^2, dm2_31^2) * U^dagger
    Matrix3cd um2uv;       // um2u * v
    Matrix3cd vum2u;       // v * um2u
    
    // m2 matrix stored as complex.
    Matrix3cd m2;
    
    // Flag to denote if matrix update is needed.
    bool need_update;
    
    // Private method: computes amplitude matrix.
    Matrix3cd _amplitude(double E, double L) {
        _set_invariants(E);
        shift_hamiltonian(E);
        shift_hamiltonian_squared(E);
        // L is rescaled by a factor 1.267
        double L_factor = L * 1.267;
        Matrix3cd result = Matrix3cd::Zero();
        
        for (int i = 0; i < 3; i++) {
            int idx1 = (i + 2) % 3;
            int idx2 = (i + 1) % 3;

            // diff_exp = (lambda[idx1] - lambda[idx2]) * exp(-1j * L_factor * lambda[i])
            complex<double> phase = exp(complex<double>(0, -L_factor * lambdas[i]));
            complex<double> diff_exp = (lambdas[idx1] - lambdas[idx2]) * phase;
            
            double l_jk = lambdas[idx1] * lambdas[idx2];
            double l_i = lambdas[i];
            
            // Add contribution: diff_exp * ((l_jk * I + l_i * Hs) + Hs2)
            Matrix3cd term = l_jk * Matrix3cd::Identity() + l_i * Hs + Hs2;
            result += diff_exp * term;
        }
        return result;
    }
    
    // Shift the Hamiltonian: Hs = H - (TrH/3) I.
    void shift_hamiltonian(double E) {
        Matrix3cd H = hamiltonian(E);
        Hs = H - (TrH / 3.0) * Matrix3cd::Identity();
    }
    
    // Shift the squared Hamiltonian.
    void shift_hamiltonian_squared(double E) {
        Matrix3cd H = hamiltonian(E);
        Matrix3cd Hsq = hamiltonian_squared(E);
        Hs2 = Hsq - (2 * TrH * H / 3.0) + (TrH * TrH / 9.0) * Matrix3cd::Identity();
    }
    
    // Compute the Hamiltonian.
    Matrix3cd hamiltonian(double E) {
        if (need_update) {
            _set_matrices();
        }
        // H = (um2u)/(2E) + v
        return um2u / (2.0 * E) + v;
    }
    
    // Compute the squared Hamiltonian.
    Matrix3cd hamiltonian_squared(double E) {
        if (need_update) {
            _set_matrices();
        }
        // H^2 = um4u/(4E^2) + v2 + (um2uv + vum2u)/(2E)
        return um4u / (4.0 * E * E) + v2 + (um2uv + vum2u) / (2.0 * E);
    }
    
    // Set invariants based on energy E.
    void _set_invariants(double E) {
        TrH = (dm2_21 + dm2_31) / (2.0 * E) + V;
        TrH2 = ( (dm2_21 * dm2_21 + dm2_31 * dm2_31) / (4.0 * E * E) )
                + V * V
                + V * ((norm(U(0, 1)) * dm2_21) + (norm(U(1, 2)) * dm2_31)) / E;
        DetH = V * dm2_21 * dm2_31 *
               norm(U(1, 1) * U(2, 2) - U(1, 2) * U(2, 1)) / ( (2.0 * E) * (2.0 * E) );
        
        TrHs2 = TrH2 - (TrH * TrH) / 3.0;
        DetHs = DetH + (TrH2 * TrH) / 6.0 - (5.0 / 54.0) * (TrH * TrH * TrH);
        Disc2_Hs = 0.5 * std::pow(TrHs2, 3) - 27 * DetHs * DetHs;
        
        // _theta = arccos(sqrt(54 * DetHs^2 / TrHs2^3))
        double ratio = std::sqrt(54.0 * DetHs * DetHs / std::pow(TrHs2, 3));

        double theta = std::acos(ratio);
        double sin_theta = std::sin(theta/3.0);
        double cos_theta = std::sqrt(1-std::pow(sin_theta, 2));
        double scale = std::sqrt(2.0 * TrHs2 / 3.0);
        lambdas[0] = scale * cos_theta;
        lambdas[1] = 0.5 * scale * (-cos_theta - std::sqrt(3.0)*sin_theta);
        lambdas[2] = -lambdas[0] - lambdas[1];
    }
    
    // Set various matrices needed for oscillation calculations.
    void _set_matrices() {
        pmns_matrix();
        // m2 diagonal: diag(0, dm2_21, dm2_31)
        m2 = Matrix3cd::Zero();
        m2(0, 0) = 0;
        m2(1, 1) = dm2_21;
        m2(2, 2) = dm2_31;
        um2u = U * m2 * U.adjoint();
        
        // um4u is U * diag(0, dm2_21^2, dm2_31^2) * U^dagger
        Matrix3cd m4 = Matrix3cd::Zero();
        m4(0, 0) = 0;
        m4(1, 1) = dm2_21 * dm2_21;
        m4(2, 2) = dm2_31 * dm2_31;
        um4u = U * m4 * U.adjoint();
        
        // v2 = diag(V^2, 0, 0)
        v2 = Matrix3cd::Zero();
        v2(0, 0) = V * V;
        // v = diag(V, 0, 0)
        v = Matrix3cd::Zero();
        v(0, 0) = V;
        
        um2uv = um2u * v;
        vum2u = v * um2u;
        
        need_update = false;
    }
    
    // Construct the PMNS matrix.
    void pmns_matrix() {
        double c12 = std::cos(theta_12);
        double s12 = std::sin(theta_12);
        double c23 = std::cos(theta_23);
        double s23 = std::sin(theta_23);
        double c13 = std::cos(theta_13);
        double s13 = std::sin(theta_13);
        // e^{-i delta_cp} since in Python we use np.conj(e^(i delta_cp)) in the (0,2) element.
        complex<double> e_idelta = std::exp(complex<double>(0, delta_cp));
        complex<double> e__idelta = std::exp(complex<double>(0, -delta_cp));
        
        // Allocate U as a 3x3 complex matrix.
        U = Matrix3cd::Zero();
        
        // First row
        U(0, 0) = c12 * c13;
        U(0, 1) = s12 * c13;
        U(0, 2) = s13 * e__idelta;
        
        // Second row
        U(1, 0) = -s12 * c23 - c12 * s23 * s13 * e_idelta;
        U(1, 1) =  c12 * c23 - s12 * s23 * s13 * e_idelta;
        U(1, 2) = s23 * c13;
        
        // Third row
        U(2, 0) = s12 * s23 - c12 * c23 * s13 * e_idelta;
        U(2, 1) = -c12 * s23 - s12 * c23 * s13 * e_idelta;
        U(2, 2) = c23 * c13;
    }
};

//
// Example main() to demonstrate usage of the CHICOS class.
//
int main() {
    const double minL = 100.0;  // km
    const double maxL = 1000.0;  // km
    const int numPoints = 100000;
    const double minEnergy = 0.0001;  // GeV
    const double maxEnergy = 10000.0;  // GeV
    
    CHICOS chicos;
    std::cout << "E(GeV),L(km),P(e->e),P(mu->e),P(tau->e),P(e->mu),P(mu->mu),P(tau->mu),P(e->tau),P(mu->tau),P(tau->tau)\n";
    
    // Process energies on the fly without storing them
    for (int i = 0; i < numPoints; i++) {
    //for (int j = 0; j < numPoints; j++) {
        double t = static_cast<double>(i) / (numPoints - 1);  // ranges from 0 to 1
        //double s = static_cast<double>(j) / (numPoints - 1);  // ranges from 0 to 1
        double energy = minEnergy * std::pow(maxEnergy/minEnergy, t);  // logarithmic spacing
        // double baseline = minL * std::pow(maxL/minL, s);  // logarithmic spacing
        double baseline = 295.0;  // logarithmic spacing
        Matrix3cd prob = chicos.compute_oscillations(energy, baseline);
                std::cout << std::scientific << std::setprecision(10)
                 << energy << ","
                 << baseline << ","
                 << std::real(prob(0,0)) << ","
                 << std::real(prob(0,1)) << ","
                 << std::real(prob(0,2)) << ","
                 << std::real(prob(1,0)) << ","
                 << std::real(prob(1,1)) << ","
                 << std::real(prob(1,2)) << ","
                 << std::real(prob(2,0)) << ","
                 << std::real(prob(2,1)) << ","
                 << std::real(prob(2,2)) << "\n";

    //}
    }
    return 0;
}
