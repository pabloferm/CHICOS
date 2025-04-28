#ifndef GCHICOS_PRECOMPUTE_H
#define GCHICOS_PRECOMPUTE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <Eigen/Dense>

class gCHICOS_Precompute {
private:
    Eigen::Matrix3cd H__10_const;
    Eigen::Matrix3cd H_00_const;
    Eigen::Matrix3cd H_01_const;
    Eigen::Matrix3cd H_10_const;
    Eigen::Matrix3cd U_const;
    Eigen::Vector3d m2_const;

    Eigen::Matrix3cd H2__20_const;
    Eigen::Matrix3cd H2__10_const;
    Eigen::Matrix3cd H2_00_const;
    Eigen::Matrix3cd H2_10_const;
    Eigen::Matrix3cd H2_20_const;
    Eigen::Matrix3cd H2__11_const;
    Eigen::Matrix3cd H2_01_const;
    Eigen::Matrix3cd H2_11_const;
    Eigen::Matrix3cd H2_02_const;

    Eigen::Matrix3cd H3__30_const;
    Eigen::Matrix3cd H3__20_const;
    Eigen::Matrix3cd H3__10_const;
    Eigen::Matrix3cd H3_00_const;
    Eigen::Matrix3cd H3_10_const;
    Eigen::Matrix3cd H3_20_const;
    Eigen::Matrix3cd H3_30_const;
    Eigen::Matrix3cd H3__21_const;
    Eigen::Matrix3cd H3__11_const;
    Eigen::Matrix3cd H3_01_const;
    Eigen::Matrix3cd H3_11_const;
    Eigen::Matrix3cd H3_21_const;
    Eigen::Matrix3cd H3__12_const;
    Eigen::Matrix3cd H3_02_const;
    Eigen::Matrix3cd H3_12_const;
    Eigen::Matrix3cd H3_03_const;

    std::complex<double> TrH__10_const;
    std::complex<double> TrH_00_const;
    std::complex<double> TrH_01_const;
    std::complex<double> TrH_10_const;

    std::complex<double> TrH2__20_const;
    std::complex<double> TrH2__10_const;
    std::complex<double> TrH2_00_const;
    std::complex<double> TrH2_10_const;
    std::complex<double> TrH2_20_const;
    std::complex<double> TrH2__11_const;
    std::complex<double> TrH2_01_const;
    std::complex<double> TrH2_11_const;
    std::complex<double> TrH2_02_const;

    std::complex<double> TrH3__30_const;
    std::complex<double> TrH3__20_const;
    std::complex<double> TrH3__10_const;
    std::complex<double> TrH3_00_const;
    std::complex<double> TrH3_10_const;
    std::complex<double> TrH3_20_const;
    std::complex<double> TrH3_30_const;
    std::complex<double> TrH3__21_const;
    std::complex<double> TrH3__11_const;
    std::complex<double> TrH3_01_const;
    std::complex<double> TrH3_11_const;
    std::complex<double> TrH3_21_const;
    std::complex<double> TrH3__12_const;
    std::complex<double> TrH3_02_const;
    std::complex<double> TrH3_12_const;
    std::complex<double> TrH3_03_const;

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
    gCHICOS_Precompute();
    void set_vacuum_hamiltonian(double theta_12 = 33.44, double theta_23 = 49.2, double theta_13 = 8.57, double delta_cp = 234.0, double dm2_21 = 7.42e-5, double dm2_31 = 2.514e-3);
    void set_matter_hamiltonian();
    void set_NSI_hamiltonian(double eps_ee = 0.0, double eps_emu = 0.0, double eps_etau = 0.0, double eps_mumu = 0.0, double eps_mutau = 0.0, double eps_tautau = 0.0);
    void set_LV_hamiltonian(double a_e = 0.0, double a_mu = 0.0, double a_tau = 0.0, double b_e = 0.0, double b_mu = 0.0, double b_tau = 0.0);
    void set_decoherence_hamiltonian(double Gamma_ee = 0.0, double Gamma_emu_re = 0.0, double Gamma_emu_im = 0.0, double Gamma_etau_re = 0.0, double Gamma_etau_im = 0.0, double Gamma_mumu = 0.0, double Gamma_mutau_re = 0.0, double Gamma_mutau_im = 0.0, double Gamma_tautau = 0.0);
    Eigen::Matrix3cd compute_oscillations(double E, double L, double rho);
    void setup_constants(); // Method to call _constant_matrices and _constant_invariants
};

#endif // GCHICOS_PRECOMPUTE_H