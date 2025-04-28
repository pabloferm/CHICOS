#ifndef GCHICOS_OPTI_H
#define GCHICOS_OPTI_H

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <Eigen/Dense>

struct HamiltonianTermsMerged {
    Eigen::Matrix3cd H__10;
    Eigen::Matrix3cd H_00;
    Eigen::Matrix3cd H_01;
    Eigen::Matrix3cd H_10;
};

struct HamiltonianSquaredTermsMerged {
    Eigen::Matrix3cd H2__20;
    Eigen::Matrix3cd H2__10;
    Eigen::Matrix3cd H2_00;
    Eigen::Matrix3cd H2_10;
    Eigen::Matrix3cd H2_20;
    Eigen::Matrix3cd H2__11;
    Eigen::Matrix3cd H2_01;
    Eigen::Matrix3cd H2_11;
    Eigen::Matrix3cd H2_02;
};

struct HamiltonianCubedTermsMerged {
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
};

struct HamiltonianTracesMerged {
    std::complex<double> TrH__10;
    std::complex<double> TrH_00;
    std::complex<double> TrH_01;
    std::complex<double> TrH_10;
};

struct HamiltonianSquaredTracesMerged {
    std::complex<double> TrH2__20;
    std::complex<double> TrH2__10;
    std::complex<double> TrH2_00;
    std::complex<double> TrH2_10;
    std::complex<double> TrH2_20;
    std::complex<double> TrH2__11;
    std::complex<double> TrH2_01;
    std::complex<double> TrH2_11;
    std::complex<double> TrH2_02;
};

struct HamiltonianCubedTracesMerged {
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
};

class gCHICOS_OptimizedMerged {
private:
    HamiltonianTermsMerged H_terms_const;
    HamiltonianSquaredTermsMerged H2_terms_const;
    HamiltonianCubedTermsMerged H3_terms_const;
    HamiltonianTracesMerged TrH_vals_const;
    HamiltonianSquaredTracesMerged TrH2_vals_const;
    HamiltonianCubedTracesMerged TrH3_vals_const;

    Eigen::Matrix3cd U_const;
    Eigen::Vector3d m2_const;

    Eigen::Matrix3cd Hs_const;
    Eigen::Matrix3cd Hs2_const;

    std::complex<double> TrHs2_const;
    std::complex<double> DetHs_const;
    double Disc2_Hs_const;
    Eigen::Vector3d lambdas_const;

    bool constants_set;

    void _construct_full_hamiltonian();
    void _constant_matrices();
    void _constant_invariants();
    void _set_shifted_hamiltonian_invariants();
    void _set_eigenvalues();

protected:
    void pmns_matrix(double theta_12, double theta_13, double theta_23, double delta_cp);

public:
    gCHICOS_OptimizedMerged();
    void set_vacuum_hamiltonian(double theta_12 = 33.44, double theta_23 = 49.2, double theta_13 = 8.57, double delta_cp = 234.0, double dm2_21 = 7.42e-5, double dm2_31 = 2.514e-3);
    void set_matter_hamiltonian();
    void set_NSI_hamiltonian(double eps_ee = 0.0, double eps_emu = 0.0, double eps_etau = 0.0, double eps_mumu = 0.0, double eps_mutau = 0.0, double eps_tautau = 0.0);
    void set_LV_hamiltonian(double a_e = 0.0, double a_mu = 0.0, double a_tau = 0.0, double b_e = 0.0, double b_mu = 0.0, double b_tau = 0.0);
    void set_decoherence_hamiltonian(double Gamma_ee = 0.0, double Gamma_emu_re = 0.0, double Gamma_emu_im = 0.0, double Gamma_etau_re = 0.0, double Gamma_etau_im = 0.0, double Gamma_mumu = 0.0, double Gamma_mutau_re = 0.0, double Gamma_mutau_im = 0.0, double Gamma_tautau = 0.0);
    void setup_constants();
    Eigen::Matrix3cd compute_oscillations(double E, double L, double rho);
};

#endif // GCHICOS_OPTI_H