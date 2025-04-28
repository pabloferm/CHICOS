#include "gchicos_precompute.h"
#include <cmath>
#include <complex>
#include <Eigen/Dense>

using namespace Eigen;

const double GF_factor = 7.56e-14;
const double SQRT3 = sqrt(3.0);

void gCHICOS_Precompute::_construct_full_hamiltonian() {
    H__10_const.setZero();
    H_00_const.setZero();
    H_01_const.setZero();
    H_10_const.setZero();
}

void gCHICOS_Precompute::_constant_matrices() {
    H2__20_const = H__10_const * H__10_const;
    H2__10_const = H__10_const * H_00_const + H_00_const * H__10_const;
    H2_00_const = H_00_const * H_00_const + H__10_const * H_10_const + H_10_const * H__10_const;
    H2_10_const = H_10_const * H_00_const + H_00_const * H_10_const;
    H2_20_const = H_10_const * H_10_const;
    H2__11_const = H__10_const * H_01_const + H_01_const * H__10_const;
    H2_01_const = H_00_const * H_01_const + H_01_const * H_00_const;
    H2_11_const = H_10_const * H_01_const + H_01_const * H_10_const;
    H2_02_const = H_01_const * H_01_const;

    H3__30_const = H__10_const * H2__20_const;
    H3__20_const = H__10_const * H2__10_const + H_00_const * H2__20_const;
    H3__10_const = H__10_const * H2_00_const + H_00_const * H2__10_const + H_10_const * H2__20_const;
    H3_00_const = H__10_const * H2_10_const + H_00_const * H2_00_const + H_10_const * H2__10_const;
    H3_10_const = H__10_const * H2_20_const + H_00_const * H2_10_const + H_10_const * H2_00_const;
    H3_20_const = H_00_const * H2_20_const + H_10_const * H2_10_const;
    H3_30_const = H_10_const * H2_20_const;
    H3__21_const = H__10_const * H2__11_const + H_01_const * H2__20_const;
    H3__11_const = H__10_const * H2_01_const + H_00_const * H2__11_const + H_01_const * H2__10_const;
    H3_01_const = H__10_const * H2_11_const + H_00_const * H2_01_const + H_01_const * H2_00_const + H_10_const * H2__11_const;
    H3_11_const = H_00_const * H2_11_const + H_01_const * H2_10_const + H_10_const * H2_01_const;
    H3_21_const = H_01_const * H2_20_const + H_10_const * H2__11_const;
    H3__12_const = H__10_const * H2_02_const + H_01_const * H2__11_const;
    H3_02_const = H_01_const * H2_01_const + H_00_const * H2_02_const;
    H3_12_const = H_01_const * H2_11_const + H_10_const * H2_02_const;
    H3_03_const = H_01_const * H2_02_const;
}

void gCHICOS_Precompute::_constant_invariants() {
    TrH__10_const = H__10_const.trace();
    TrH_00_const = H_00_const.trace();
    TrH_01_const = H_01_const.trace();
    TrH_10_const = H_10_const.trace();

    TrH2__20_const = H2__20_const.trace();
    TrH2__10_const = H2__10_const.trace();
    TrH2_00_const = H2_00_const.trace();
    TrH2_10_const = H2_10_const.trace();
    TrH2_20_const = H2_20_const.trace();
    TrH2__11_const = H2__11_const.trace();
    TrH2_01_const = H2_01_const.trace();
    TrH2_11_const = H2_11_const.trace();
    TrH2_02_const = H2_02_const.trace();

    TrH3__30_const = H3__30_const.trace();
    TrH3__20_const = H3__20_const.trace();
    TrH3__10_const = H3__10_const.trace();
    TrH3_00_const = H3_00_const.trace();
    TrH3_10_const = H3_10_const.trace();
    TrH3_20_const = H3_20_const.trace();
    TrH3_30_const = H3_30_const.trace();
    TrH3__21_const = H3__21_const.trace();
    TrH3__11_const = H3__11_const.trace();
    TrH3_01_const = H3_01_const.trace();
    TrH3_11_const = H3_11_const.trace();
    TrH3_21_const = H3_21_const.trace();
    TrH3__12_const = H3__12_const.trace();
    TrH3_02_const = H3_02_const.trace();
    TrH3_12_const = H3_12_const.trace();
    TrH3_03_const = H3_03_const.trace();
}

gCHICOS_Precompute::gCHICOS_Precompute() {
    _construct_full_hamiltonian();
    need_update = true;
}

void gCHICOS_Precompute::setup_constants() {
    _constant_matrices();
    _constant_invariants();
}

// ... (Implement other methods like set_vacuum_hamiltonian, compute_oscillations, etc.,
// using the _const member variables where appropriate)