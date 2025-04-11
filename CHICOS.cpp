#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <iomanip>

using namespace Eigen;
using namespace std;

class CHICOS {
private:
    Matrix3cd U, Hs, Hs2, v, v2, um2u, um4u, um2uv, vum2u, m2;
    Vector3cd lambdas;
    double theta_12, theta_23, theta_13, delta_cp, dm2_21, dm2_31, V;
    double TrH, TrH2, DetH, TrHs2, DetHs, Disc2_Hs;
    bool need_update;
    
    // Cached values to avoid recalculation
    double c12, s12, c23, s23, c13, s13;

public:
    CHICOS(double theta_12 = 33.44, double theta_23 = 49.2, double theta_13 = 8.57,
           double delta_cp = 234, double dm2_21 = 7.42e-5, double dm2_31 = 2.514e-3,
           double density = 2.8) 
        : U(Matrix3cd::Zero()), 
          Hs(Matrix3cd::Zero()), 
          Hs2(Matrix3cd::Zero()), 
          v(Matrix3cd::Zero()), 
          v2(Matrix3cd::Zero()), 
          um2u(Matrix3cd::Zero()), 
          um4u(Matrix3cd::Zero()), 
          um2uv(Matrix3cd::Zero()), 
          vum2u(Matrix3cd::Zero()), 
          m2(Matrix3cd::Zero()),
          lambdas(Vector3cd::Zero()),
          V(7.56e-14 * 0.5 * density),
          theta_12(M_PI * theta_12 / 180.0),
          theta_23(M_PI * theta_23 / 180.0),
          theta_13(M_PI * theta_13 / 180.0),
          delta_cp(M_PI * delta_cp / 180.0),
          dm2_21(dm2_21),
          dm2_31(dm2_31),
          need_update(true)
    {
        _set_matrices();
    }

    // Compute individual oscillation
    Matrix3cd compute_oscillations(const double E, const double L) {
        if (need_update) _set_matrices();
        _set_invariants(E);
        Matrix3cd J = _amplitude(E, L);
        double denom = (Disc2_Hs != 0.0) ? Disc2_Hs : 1.0;
        return J.cwiseAbs2() / denom;
    }

    // Calculate amplitude for oscillations
    Matrix3cd _amplitude(const double E, double L) const {
        L *= 1.267;
        Matrix3cd result = Matrix3cd::Zero();
        for (int i = 0; i < 3; ++i) {
            int j = (i + 2) % 3;  // i-1 wrapped
            int k = (i + 1) % 3;  // i-2 wrapped
            complex<double> diff_exp = (lambdas[j] - lambdas[k]) * exp(complex<double>(0.0, -1.0) * L * lambdas[i]);
            complex<double> l_jk = lambdas[j] * lambdas[k];
            complex<double> l_i = lambdas[i];
            // Combined operation to reduce temporary matrices
            result += diff_exp * (l_jk * Matrix3cd::Identity() + l_i * Hs + Hs2);
        }
        return result;
    }

    void _set_matrices() {
        pmns_matrix();
        
        // Set m2 directly without initializing to zero first
        m2.setZero();
        m2(1, 1) = dm2_21;
        m2(2, 2) = dm2_31;

        const Matrix3cd& U_conj_T = U.adjoint();
        um2u = U * m2 * U_conj_T;

        // Calculate um4u without creating a temporary m4 matrix
        um4u.setZero();
        um4u += U * (dm2_21 * dm2_21 * (U_conj_T.col(1) * U_conj_T.row(1)));
        um4u += U * (dm2_31 * dm2_31 * (U_conj_T.col(2) * U_conj_T.row(2)));

        // Directly set v and v2 values without initializing
        v.setZero();
        v(0, 0) = V;

        v2.setZero();
        v2(0, 0) = V * V;

        um2uv = um2u * v;
        vum2u = v * um2u;
        need_update = false;
    }
    void pmns_matrix() {
        // Cache these values to avoid recalculation
        c12 = cos(theta_12), s12 = sin(theta_12);
        c23 = cos(theta_23), s23 = sin(theta_23);
        c13 = cos(theta_13), s13 = sin(theta_13);
        complex<double> e_idelta = exp(complex<double>(0, delta_cp));
        complex<double> ce_idelta = exp(complex<double>(0, -delta_cp));

        U << c12 * c13, s12 * c13, s13 * ce_idelta,
            -s12 * c23 - c12 * s23 * s13 * e_idelta, c12 * c23 - s12 * s23 * s13 * e_idelta, s23 * c13,
            s12 * s23 - c12 * c23 * s13 * e_idelta, -c12 * s23 - s12 * c23 * s13 * e_idelta, c23 * c13;
    }

    // Calculate Hs and Hs2 directly to avoid redundant calls to hamiltonian() methods
    void calculate_hamiltonians(const double E) {
        double inv_2E = 1.0 / (2.0 * E);
        double inv_4E2 = inv_2E * inv_2E;
        
        // Calculate H = um2u/(2E) + v directly
        Matrix3cd H = um2u * inv_2E + v;
        
        // Calculate Hs = H - (TrH/3)*I
        Hs = H - (TrH / 3.0) * Matrix3cd::Identity();
        
        // Calculate H² directly
        Matrix3cd H2 = um4u * inv_4E2 + v2 + (um2uv + vum2u) * inv_2E;
        
        // Calculate Hs² = H² - 2(TrH/3)*H + (TrH²/9)*I
        Hs2 = H2 - (2.0 * TrH / 3.0) * H + (TrH * TrH / 9.0) * Matrix3cd::Identity();
    }
    
    Matrix3cd hamiltonian(const double E) const {
        return um2u / (2.0 * E) + v;
    }

    Matrix3cd hamiltonian_squared(const double E) const {
        return um4u / (4.0 * E * E) + v2 + (um2uv + vum2u) / (2.0 * E);
    }

    void _set_invariants(const double E) {
        TrH = (dm2_21 + dm2_31) / (2 * E) + V;
        TrH2 = (dm2_21 * dm2_21 + dm2_31 * dm2_31) / (4 * E * E) + V * V + 
               V * (std::abs(U(0, 1)) * std::abs(U(0, 1)) * dm2_21 + 
                   std::abs(U(1, 2)) * std::abs(U(1, 2)) * dm2_31) / E;
        complex<double> u_det = U(1, 1) * U(2, 2) - U(1, 2) * U(2, 1);
        DetH = V * dm2_21 * dm2_31 * std::abs(u_det) * std::abs(u_det) / (4 * E * E);
        TrHs2 = TrH2 - TrH * TrH / 3.0;
        DetHs = DetH + TrH2 * TrH / 6.0 - 5.0 * TrH * TrH * TrH / 54.0;
        Disc2_Hs = 0.5 * std::pow(TrHs2, 3) - 27.0 * DetHs * DetHs;

        double theta = std::acos(std::sqrt(54.0 * DetHs * DetHs / std::pow(TrHs2, 3)));
        double sin_theta = std::sin(theta/3.0);
        double cos_theta = std::sqrt(1-std::pow(sin_theta, 2));
        double scale = std::sqrt(2.0 * TrHs2 / 3.0);
        lambdas[0] = scale * cos_theta;
        lambdas[1] = 0.5 * scale * (-cos_theta - std::sqrt(3.0)*sin_theta);
        lambdas[2] = -lambdas[0] - lambdas[1];
    }
};


int main() {
    const double L = 295.0;  // km
    const int numPoints = 1000000;
    const double minEnergy = 0.01;  // GeV
    const double maxEnergy = 1000.0;  // GeV
    
    CHICOS chicos;
    std::cout << "E(GeV)\tP(e->e)\tP(mu->e)\tP(tau->e)\n";
    
    // Process energies on the fly without storing them
    for (int i = 0; i < numPoints; i++) {
        double t = static_cast<double>(i) / (numPoints - 1);  // ranges from 0 to 1
        double energy = minEnergy * std::pow(maxEnergy/minEnergy, t);  // logarithmic spacing
        Matrix3cd prob = chicos.compute_oscillations(energy, L);
/*                std::cout << std::scientific << std::setprecision(6)
                 << energy << "\t"
                 << std::real(prob(0,0)) << "\t"
                 << std::real(prob(0,1)) << "\t"
                 << std::real(prob(0,2)) << "\n";
*/
    }
    return 0;
}
