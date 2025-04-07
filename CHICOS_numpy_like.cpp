#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Array>
#include <complex>
#include <cmath>

using namespace Eigen;
using namespace std;

class CHICOS {
private:
    const double G_F = 1.1663787e-5;
    const double Y_e = 0.5;
    double V;
    double theta_12, theta_23, theta_13, delta_cp;
    double dm2_21, dm2_31;
    Matrix3cd U;
    Matrix3cd um2u, um4u, v, v2, um2uv, I;
    ArrayXXcd lambdas;
    ArrayXd Disc2_Hs;
    bool need_update;

public:
    CHICOS(double theta12 = 33.44, double theta23 = 49.2, double theta13 = 8.57, double deltaCP = 234,
           double dm221 = 7.42e-5, double dm231 = 2.514e-3, double density = 2.8)
        : dm2_21(dm221), dm2_31(dm231), need_update(true) {
        V = 7.56e-14 * Y_e * density;
        theta_12 = theta12 * M_PI / 180;
        theta_23 = theta23 * M_PI / 180;
        theta_13 = theta13 * M_PI / 180;
        delta_cp = deltaCP * M_PI / 180;
        I = Matrix3cd::Identity();
        set_matrices();
    }

    void set_matrices() {
        double c12 = cos(theta_12), s12 = sin(theta_12);
        double c23 = cos(theta_23), s23 = sin(theta_23);
        double c13 = cos(theta_13), s13 = sin(theta_13);
        complex<double> e_idelta = polar(1.0, delta_cp);

        U << c12 * c13, s12 * c13, s13 * conj(e_idelta),
            -s12 * c23 - c12 * s23 * s13 * e_idelta, c12 * c23 - s12 * s23 * s13 * e_idelta, s23 * c13,
            s12 * s23 - c12 * c23 * s13 * e_idelta, -c12 * s23 - s12 * c23 * s13 * e_idelta, c23 * c13;

        Matrix3cd m2;
        m2 << 0, 0, 0,
              0, dm2_21, 0,
              0, 0, dm2_31;

        um2u = U * m2 * U.adjoint();
        um4u = U * m2.array().square().matrix() * U.adjoint();
        v = Matrix3cd::Zero();
        v(0, 0) = V;
        v2 = v * v;
        um2uv = um2u * v;
        need_update = false;
    }

    void set_invariants(const ArrayXd& E) {
        int N = E.size();
        lambdas.resize(3, N);
        Disc2_Hs.resize(N);

        for (int i = 0; i < N; ++i) {
            Matrix3cd Hs = um2u / (2.0 * E(i)) + v - ((um2u / (2.0 * E(i)) + v).trace() / 3.0) * I;
            Matrix3cd Hs2 = (um4u / (4.0 * E(i) * E(i))) + v2 + um2uv / E(i) - 2.0 * (Hs.trace() / 3.0) * Hs + (pow(Hs.trace(), 2) / 9.0) * I;
            double TrHs2 = real(Hs2.trace());
            double DetHs = real(Hs.determinant()) + (pow(TrHs2, 1.5) / 6.0) - (5.0 / 54.0) * pow(real(Hs.trace()), 3);
            Disc2_Hs(i) = 0.5 * pow(TrHs2, 3) - 27.0 * pow(DetHs, 2);
            double theta = acos(sqrt(54.0 * pow(DetHs, 2) / pow(TrHs2, 3)));
            double scale = sqrt(2.0 * TrHs2 / 3.0);
            for (int k = 0; k < 3; ++k) {
                lambdas(k, i) = scale * cos((theta + 2.0 * M_PI * k) / 3.0);
            }
        }
    }

    MatrixXcd oscillator(const ArrayXd& E, const ArrayXd& L) {
        int N = E.size();
        MatrixXcd result = MatrixXcd::Zero(3, N);
        set_invariants(E);
        ArrayXd L_scaled = L * 1.267;
        
        for (int i = 0; i < N; ++i) {
            Matrix3cd temp = Matrix3cd::Zero();
            for (int j = 0; j < 3; ++j) {
                complex<double> diff_exp = (lambdas((j + 1) % 3, i) - lambdas((j + 2) % 3, i)) * exp(-1i * L_scaled(i) * lambdas(j, i));
                complex<double> l_jk = lambdas((j + 1) % 3, i) * lambdas((j + 2) % 3, i);
                complex<double> l_i = lambdas(j, i);
                temp += diff_exp * ((l_jk * I + l_i * um2u) + um4u);
            }
            result.col(i) = temp.diagonal() / Disc2_Hs(i);
        }
        return result;
    }
};

int main() {
    CHICOS model;
    ArrayXd E = ArrayXd::LinSpaced(1000, 1.0, 10.0);  // 1000 energy values from 1 to 10 GeV
    ArrayXd L = ArrayXd::Constant(1000, 500);  // Constant baseline of 500 km
    cout << "Oscillation probabilities computed!" << endl;
    return 0;
}
