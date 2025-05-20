#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace Eigen;
using namespace std;

const double PI = std::acos(-1);
const double WEAK = 7.632470714045e-5;
const double BASELINE_FACTOR = 5.067730716156394;
const double SQRT_3 = std::sqrt(3);

class CHICOS {
public:
    // Constructor with default parameters (angles in degrees)
    CHICOS(double theta_12_deg = 33.44,
           double theta_23_deg = 49.2,
           double theta_13_deg = 8.57,
           double delta_cp_deg = 234,
           double dm2_21 = 7.42e-5,
           double dm2_31 = 2.51e-3,
           double density = 2.8,   // g/cm^3
           double Y_e = 0.5    // unitless (effective electron fraction)
           )
    {
        // Oscillation parameters
        theta_12 = theta_12_deg * PI / 180.0;
        theta_23 = theta_23_deg * PI / 180.0;
        theta_13 = theta_13_deg * PI / 180.0;
        delta_cp = delta_cp_deg * PI / 180.0;
        this->dm2_21 = dm2_21;
        this->dm2_31 = dm2_31;
        
        // Matter effects
        V = WEAK * Y_e * density;
        
        // Initiate math
        need_update = true;
        _set_matrices();
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
                prob(i, j) = std::norm(J(i, j));
            }
        }
        return prob / Disc2_Hs;
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
    Vector3cd exp_lambdas;
    
    // Pre-computed matrices (all 3x3 matrices)
    Matrix3cd U;           // PMNS matrix
    Matrix3cd H;          // Hamiltonian
    Matrix3cd H2;          // Hamiltonian squared
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
        //_set_invariants(E);
        _set_invariants_hamiltonians(E);
        double L_factor = L * BASELINE_FACTOR;

        Matrix3cd result = Matrix3cd::Zero();
        Matrix3cd term;
        complex<double> diff_exp, exp_lambda;

        exp_lambdas[0] = exp(complex<double>(0, -L_factor * lambdas[0]));
        exp_lambdas[1] = exp(complex<double>(0, -L_factor * lambdas[1]));
        exp_lambdas[2] = 1.0 / (exp_lambdas[0] * exp_lambdas[1]);

        for (int i = 0; i < 3; i++) {
            int k = (i + 2) % 3;
            int j = (i + 1) % 3;
            // diff_exp = (lambda[idx1] - lambda[idx2]) * exp(-1j * L_factor * lambda[i])
            diff_exp = (lambdas[k] - lambdas[j]) * exp_lambdas[i];
            // Matrix combination: l_j*l_k * I + l_i * Hs + Hs2
            term = lambdas[j] * lambdas[k] * Matrix3cd::Identity() + lambdas[i] * Hs + Hs2;
            result += diff_exp * term;
        }
        return result;
    }
    
    void _set_invariants_hamiltonians(double E) {
        if (need_update) _set_matrices();

        // H = (um2u)/(2E) + v
        H = um2u / (2.0 * E) + v;        
        // H^2 = um4u/(4E^2) + v2 + (um2uv + vum2u)/(2E)
        H2 = um4u / (4.0 * E * E) + v2 + (um2uv + vum2u) / (2.0 * E);
        // Trace of Hamiltonian
        TrH = std::real(H(0,0) + H(1,1) + H(2,2));
        // Trace of Hamiltonian squared
        TrH2 = std::real(H2(0,0) + H2(1,1) + H2(2,2));
        // Determinant of Hamiltonian
        DetH = V * dm2_21 * dm2_31 * std::pow(std::abs(U(1, 1) * U(2, 2) - U(1, 2) * U(2, 1)), 2) / std::pow((2.0 * E), 2);

        // Shift the Hamiltonian: Hs = H - (TrH/3) I.
        Hs = H - (TrH / 3.0) * Matrix3cd::Identity();
        // Shift the squared Hamiltonian.
        Hs2 = H2 - (2 * TrH * H / 3.0) + (TrH * TrH / 9.0) * Matrix3cd::Identity();
        // Trace of shifted Hamiltonian squared
        TrHs2 = std::real(Hs2(0,0) + Hs2(1,1) + Hs2(2,2));
        // Determinant of shifted Hamiltonian
        DetHs = DetH + (TrH2 * TrH) / 6.0 - (5.0 / 54.0) * std::pow(TrH, 3);
        // Disciminant of (shifted) Hamiltonian
        Disc2_Hs = std::real(0.5 * std::pow(TrHs2, 3) - 27 * DetHs * DetHs);

        // Eigenvalues of shifted Hamiltonian
        // _theta = arccos(sqrt(54 * DetHs^2 / TrHs2^3))
        double ratio = std::sqrt(54.0 * DetHs * DetHs / std::pow(TrHs2, 3));
        double theta = std::acos(ratio);
        double sin_theta = std::sin(theta/3.0);
        double cos_theta = std::sqrt(1-std::pow(sin_theta, 2));
        double scale = std::sqrt(2.0 * TrHs2 / 3.0);
        lambdas[0] = scale * cos_theta;
        lambdas[1] = 0.5 * scale * (-cos_theta - SQRT_3*sin_theta);
        lambdas[2] = - lambdas[0] - lambdas[1];
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
        double s12 = std::sin(theta_12);
        double c12 = std::sqrt(1-std::pow(s12, 2));
        double s23 = std::sin(theta_23);
        double c23 = std::sqrt(1-std::pow(s23, 2));
        double s13 = std::sin(theta_13);
        double c13 = std::sqrt(1-std::pow(s13, 2));
        complex<double> e_idelta = std::exp(complex<double>(0, delta_cp));
        complex<double> e_midelta = 1.0 / e_idelta;
        
        // Allocate U as a 3x3 complex matrix.
        U = Matrix3cd::Zero();
        
        // First row
        U(0, 0) = c12 * c13;
        U(0, 1) = s12 * c13;
        U(0, 2) = s13 * e_midelta;
        
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
    using namespace std::chrono;
    long long total_duration_ns = 0;
    size_t count = 0;

    const double minL = 10.0;  // km
    const double maxL = 10000.0;  // km
    const int numPoints = 100000;
    const double minEnergy = 0.05;  // GeV
    const double maxEnergy = 5.0;  // GeV
    
    CHICOS chicos;
    //std::cout << "E(GeV),L(km),P(e->e),P(mu->e),P(tau->e),P(e->mu),P(mu->mu),P(tau->mu),P(e->tau),P(mu->tau),P(tau->tau)\n";
    
    // Process energies on the fly without storing them
    for (int i = 0; i < numPoints; i++) {
    //for (int j = 0; j < numPoints; j++) {
        double t = static_cast<double>(i) / (numPoints - 1);  // ranges from 0 to 1
        //double s = static_cast<double>(j) / (numPoints - 1);  // ranges from 0 to 1
        double energy = minEnergy * std::pow(maxEnergy/minEnergy, t);  // logarithmic spacing
        //double baseline = minL * std::pow(maxL/minL, s);  // logarithmic spacing
        double baseline = 300.0;  // logarithmic spacing
        auto start = high_resolution_clock::now();
        Matrix3cd prob = chicos.compute_oscillations(energy, baseline);
        auto end = high_resolution_clock::now();
        total_duration_ns += duration_cast<nanoseconds>(end - start).count();
        ++count;
        
                std::cout << std::scientific << std::setprecision(7)
                 << energy << ","
                 << baseline << ","
                // << std::real(prob(0,0)) << ","
                 << std::real(prob(0,1)) << ","
                // << std::real(prob(0,2)) << ","
                // << std::real(prob(1,0)) << ","
                 << std::real(prob(1,1)) << ","
                // << std::real(prob(1,2)) << ","
                // << std::real(prob(2,0)) << ","
                 << std::real(prob(2,1)) << "\n";
                // << std::real(prob(2,2)) << "\n";
        

    //}
    }
    double avg_ns = static_cast<double>(total_duration_ns) / count / 9.0;
    std::cout << "Total calls: " << count << "\n";
    std::cout << "Average time per call: " << avg_ns << " ns" << std::endl;
    return 0;
}
