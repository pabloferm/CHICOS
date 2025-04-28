#ifndef GCHICOS_H
#define GCHICOS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <Eigen/Dense>

class gCHICOS_Optimized {
private:
    Eigen::Matrix3cd H__10;
    Eigen::Matrix3cd H_00;
    Eigen::Matrix3cd H_01;
    Eigen::Matrix3cd H_10;
    Eigen::Matrix3cd U;
    Eigen::Vector3d m2;

    // Precomputed terms for Hamiltonian squared
    Eigen::Matrix3cd H2__20;
    Eigen::Matrix3cd H2__10;
    Eigen::Matrix3cd H2_00;
    Eigen::Matrix3cd H2_10;
    Eigen::Matrix3cd H2_20;
    Eigen::Matrix3cd H2__11;
    Eigen::Matrix3cd H2_01;
    Eigen::Matrix3cd H2_11;
    Eigen::Matrix3cd H2_02;

    // Precomputed terms for Hamiltonian cubed
    Eigen::Matrix3cd H3__30;
    Eigen::Matrix3cd H3__20;
    Eigen::Matrix3cd H3__10;
    Eigen::Matrix3cd H3_00;
    Eigen::Matrix3cd H3_10;
    Eigen::Matrix3cd H3_20;
    Eigen::Matrix3cd H3_30;
    Eigen::Matrix3cd H3__21;
    Eigen::Matrix3cd H3__11;
    Eigen::Matrix3cd H3_01;
    Eigen::Matrix3cd H3_11;
    Eigen::Matrix3cd H3_21;
    Eigen::Matrix3cd H3__12;
    Eigen::Matrix3cd H3_02;
    Eigen::Matrix3cd H3_12;
    Eigen::Matrix3cd H3_03;

    // Precomputed traces
    std::complex<double> TrH__10;
    std::complex<double> TrH_00;
    std::complex<double> TrH_01;
    std::complex<double> TrH_10;
    std::complex<double> TrH2__20;
    std::complex<double> TrH2__10;
    std::complex<double> TrH2_00;
    std::complex<double> TrH2_10;
    std::complex<double> TrH2_20;
    std::complex<double> TrH2__11;
    std::complex<double> TrH2_01;
    std::complex<double> TrH2_11;
    std::complex<double> TrH2_02;
    std::complex<double> TrH3__30;
    std::complex<double> TrH3__20;
    std::complex<double> TrH3__10;
    std::complex<double> TrH3_00;
    std::complex<double> TrH3_10;
    std::complex<double> TrH3_20;
    std::complex<double> TrH3_30;
    std::complex<double> TrH3__21;
    std::complex<double> TrH3__11;
    std::complex<double> TrH3_01;
    std::complex<double> TrH3_11;
    std::complex<double> TrH3_21;
    std::complex<double> TrH3__12;
    std::complex<double> TrH3_02;
    std::complex<double> TrH3_12;
    std::complex<double> TrH3_03;

    Eigen::Matrix3cd H;
    Eigen::Matrix3cd H2;
    Eigen::Matrix3cd H3;
    Eigen::Matrix3cd Hs;
    Eigen::Matrix3cd Hs2;

    std::complex<double> TrH;
    std::complex<double> TrH2;
    std::complex<double> DetH;
    std::complex<double> TrHs2_invariant;
    std::complex<double> DetHs;
    double Disc2_Hs;
    Eigen::Vector3d lambdas;

    bool need_update;

    void _construct_full_hamiltonian();
    void _constant_matrices();
    void _constant_invariants();
    void _set_matrices(double E, double rho);
    void _set_invariants(double E, double rho);
    Eigen::Matrix3cd _amplitude(double E, double L, double rho);

protected:
    void pmns_matrix(double theta_12, double theta_13, double theta_23, double delta_cp);
    void setup_physics(double E, double rho);

public:
    gCHICOS_Optimized();
    void set_vacuum_hamiltonian(double theta_12 = 33.44, double theta_23 = 49.2, double theta_13 = 8.57, double delta_cp = 234.0, double dm2_21 = 7.42e-5, double dm2_31 = 2.514e-3);
    void set_matter_hamiltonian();
    void set_NSI_hamiltonian(double eps_ee = 0.0, double eps_emu = 0.0, double eps_etau = 0.0, double eps_mumu = 0.0, double eps_mutau = 0.0, double eps_tautau = 0.0);
    void set_LV_hamiltonian(double a_e = 0.0, double a_mu = 0.0, double a_tau = 0.0, double b_e = 0.0, double b_mu = 0.0, double b_tau = 0.0);
    void set_decoherence_hamiltonian(double Gamma_ee = 0.0, double Gamma_emu_re = 0.0, double Gamma_emu_im = 0.0, double Gamma_etau_re = 0.0, double Gamma_etau_im = 0.0, double Gamma_mumu = 0.0, double Gamma_mutau_re = 0.0, double Gamma_mutau_im = 0.0, double Gamma_tautau = 0.0);
    Eigen::Matrix3cd compute_oscillations(double E, double L, double rho);
    void setup_constants(); // Method to call _constant_matrices and _constant_invariants
};

#endif // GCHICOS_H