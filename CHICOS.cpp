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

public:
    CHICOS(double theta_12 = 33.44, double theta_23 = 49.2, double theta_13 = 8.57,
           double delta_cp = 234, double dm2_21 = 7.42e-5, double dm2_31 = 2.514e-3,
           double density = 2.8) {
        V = 7.56e-14 * 0.5 * density;
        this->theta_12 = M_PI * theta_12 / 180.0;
        this->theta_23 = M_PI * theta_23 / 180.0;
        this->theta_13 = M_PI * theta_13 / 180.0;
        this->delta_cp = M_PI * delta_cp / 180.0;
        this->dm2_21 = dm2_21;
        this->dm2_31 = dm2_31;
        need_update = true;
        lambdas = Vector3cd::Zero();  // Initialize lambdas vector
        _set_matrices();
    }

    /* Method disabled as it expects a scalar probability but we now return a matrix
    vector<double> oscillator(const vector<double>& E_array, const vector<double>& L_array) {
        vector<double> results;
        for (size_t i = 0; i < E_array.size(); ++i) {
            results.push_back(compute_oscillations(E_array[i], L_array[i]));
        }
        return results;
    }
    */
    // Compute individual oscillation
    Matrix3cd compute_oscillations(double E, double L) {
        _set_invariants(E);
        shift_hamiltonian(E);
        shift_hamiltonian_squared(E);
        Matrix3cd J = _amplitude(E, L);
        double denom = (Disc2_Hs != 0.0) ? Disc2_Hs : 1.0;
        return J.cwiseAbs2() / denom;
    }

    // Calculate amplitude for oscillations
    Matrix3cd _amplitude(double E, double L) {
        L *= 1.267;
        Matrix3cd result = Matrix3cd::Zero();
        for (int i = 0; i < 3; ++i) {
            int j = (i + 2) % 3;  // i-1 wrapped
            int k = (i + 1) % 3;  // i-2 wrapped
            complex<double> diff_exp = (lambdas[j] - lambdas[k]) * exp(complex<double>(0.0, -1.0) * L * lambdas[i]);
            complex<double> l_jk = lambdas[j] * lambdas[k];
            complex<double> l_i = lambdas[i];
            result += diff_exp * ((l_jk * Matrix3cd::Identity() + l_i * Hs) + Hs2);
        }
        return result;
    }

    void _set_matrices() {
        pmns_matrix();
        m2 = Matrix3cd::Zero();
        m2(1, 1) = dm2_21;
        m2(2, 2) = dm2_31;

        Matrix3cd U_conj_T = U.adjoint();
        um2u = U * m2 * U_conj_T;

        Matrix3cd m4 = Matrix3cd::Zero();
        m4(1, 1) = dm2_21 * dm2_21;
        m4(2, 2) = dm2_31 * dm2_31;
        um4u = U * m4 * U_conj_T;

        v = Matrix3cd::Zero();
        v(0, 0) = V;

        v2 = Matrix3cd::Zero();
        v2(0, 0) = V * V;

        um2uv = um2u * v;
        vum2u = v * um2u;
        need_update = false;
    }
    void pmns_matrix() {
        double c12 = cos(theta_12), s12 = sin(theta_12);
        double c23 = cos(theta_23), s23 = sin(theta_23);
        double c13 = cos(theta_13), s13 = sin(theta_13);
        complex<double> e_idelta = exp(complex<double>(0, delta_cp));
        complex<double> ce_idelta = exp(complex<double>(0, -delta_cp));

        U << c12 * c13, s12 * c13, s13 * ce_idelta,
            -s12 * c23 - c12 * s23 * s13 * e_idelta, c12 * c23 - s12 * s23 * s13 * e_idelta, s23 * c13,
            s12 * s23 - c12 * c23 * s13 * e_idelta, -c12 * s23 - s12 * c23 * s13 * e_idelta, c23 * c13;
    }

    void shift_hamiltonian(double E) {
        Hs = hamiltonian(E) - (TrH / 3.0) * Matrix3cd::Identity();
    }

    void shift_hamiltonian_squared(double E) {
        Hs2 = hamiltonian_squared(E) - 
              2.0 * (TrH / 3.0) * hamiltonian(E) + 
              (TrH * TrH / 9.0) * Matrix3cd::Identity();
    }
    Matrix3cd hamiltonian(double E) {
        if (need_update) _set_matrices();
        return um2u / (2.0 * E) + v;
    }

    Matrix3cd hamiltonian_squared(double E) {
        if (need_update) _set_matrices();
        return um4u / (4.0 * E * E) + v2 + (um2uv + vum2u) / (2.0 * E);
    }

    void _set_invariants(double E) {
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
        double scale = std::sqrt(2.0 * TrHs2 / 3.0);
        for (int k = 0; k < 3; ++k) {
            lambdas[k] = scale * std::cos((theta + 2.0 * M_PI * k) / 3.0);
        }
    }
};


/*
int main() {
    const double L = 295.0;  // km
    std::vector<double> E;
    
    // Generate logarithmically spaced energies
    for (int i = 0; i < 1000000; i++) {
        double t = i / 999999.0;  // ranges from 0 to 1
        E.push_back(0.01 * std::pow(100000.0, t));  // 0.01 to 10 GeV
    }

    CHICOS chicos;
    std::cout << "E(GeV)\tP(e->e)\tP(mu->e)\tP(tau->e)\n";
    for (double energy : E) {
        Matrix3cd prob = chicos.compute_oscillations(energy, L)
                std::cout << std::scientific << std::setprecision(6)
                 << energy << "\t"
                 << std::real(prob(0,0)) << "\t"
                 << std::real(prob(0,1)) << "\t"
                 << std::real(prob(0,2)) << "\n";
    }
    return 0;
}
*/