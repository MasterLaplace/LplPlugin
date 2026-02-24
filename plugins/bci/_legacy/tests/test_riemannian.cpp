// File: test_riemannian.cpp
// Description: Tests unitaires pour RiemannianGeometry (covariance, Jacobi, δ_R, D_M)
// Build: g++ -std=c++20 -I../include -I../../shared -o test_riemannian test_riemannian.cpp && ./test_riemannian
// Auteur: MasterLaplace

#include "RiemannianGeometry.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace RiemannianGeometry;

static bool near(float a, float b, float eps = 1e-3f) { return std::abs(a - b) < eps; }

// ─── Tests jacobi_eigen ────────────────────────────────────────────────────────

static void test_jacobi_2x2_known()
{
    // M = [[3, 1], [1, 3]] → eigenvalues : 4 et 2
    Matrix M = {
        {3.0f, 1.0f},
        {1.0f, 3.0f}
    };
    std::vector<float> evals;
    Matrix V;
    jacobi_eigen(M, evals, V);

    // Trier pour comparaison
    std::sort(evals.begin(), evals.end());
    assert(near(evals[0], 2.0f));
    assert(near(evals[1], 4.0f));
    printf("[PASS] jacobi_eigen([[3,1],[1,3]]) : λ=[%.3f, %.3f]  (expected [2.0, 4.0])\n", evals[0], evals[1]);
}

static void test_jacobi_identity()
{
    // Matrice identité 4×4 → toutes les valeurs propres = 1
    Matrix M = identity(4);
    std::vector<float> evals;
    Matrix V;
    jacobi_eigen(M, evals, V);

    for (float lam : evals)
        assert(near(lam, 1.0f));
    printf("[PASS] jacobi_eigen(I_4) : toutes les valeurs propres ≈ 1.0\n");
}

static void test_jacobi_diagonal()
{
    // Matrice diagonale → eigenvalues = diagonale
    Matrix M = zeros(3);
    M[0][0] = 5.0f;
    M[1][1] = 2.0f;
    M[2][2] = 8.0f;
    std::vector<float> evals;
    Matrix V;
    jacobi_eigen(M, evals, V);

    std::sort(evals.begin(), evals.end());
    assert(near(evals[0], 2.0f));
    assert(near(evals[1], 5.0f));
    assert(near(evals[2], 8.0f));
    printf("[PASS] jacobi_eigen(diag(5,2,8)) : λ=[%.3f, %.3f, %.3f]  (expected [2,5,8])\n", evals[0], evals[1],
           evals[2]);
}

// ─── Tests compute_covariance ─────────────────────────────────────────────────

static void test_covariance_2d_known()
{
    // Données: [[1,1],[2,2],[3,3]] → covariance = [[1,1],[1,1]]
    std::vector<std::vector<float>> window = {
        {1.0f, 1.0f},
        {2.0f, 2.0f},
        {3.0f, 3.0f}
    };
    Matrix C = compute_covariance(window);
    // Var = E[(x-2)²] = ((1-2)²+(2-2)²+(3-2)²) / 2 = 2/2 = 1.0
    assert(near(C[0][0], 1.0f));
    assert(near(C[1][1], 1.0f));
    assert(near(C[0][1], 1.0f));
    assert(near(C[1][0], 1.0f));
    printf("[PASS] compute_covariance([[1,1],[2,2],[3,3]]) = [[1,1],[1,1]]\n");
}

static void test_covariance_uncorrelated()
{
    // Canal 0 et canal 1 décorrélés : covariance hors-diagonale ≈ 0
    std::vector<std::vector<float>> window = {
        {1.0f,  -1.0f},
        {-1.0f, 1.0f },
        {1.0f,  -1.0f},
        {-1.0f, 1.0f }
    };
    Matrix C = compute_covariance(window);
    // cov(X,Y) < 0 ici (corrélation négative parfaite)
    assert(C[0][1] < 0.0f);
    printf("[PASS] compute_covariance(anti-correle) : cov[0][1]=%.3f < 0\n", C[0][1]);
}

// ─── Tests riemannian_distance ────────────────────────────────────────────────

static void test_riemannian_self_distance()
{
    // δ_R(C, C) = 0 pour toute matrice SPD
    Matrix C = {
        {4.0f, 1.0f},
        {1.0f, 3.0f}
    };
    float d = riemannian_distance(C, C);
    assert(near(d, 0.0f, 1e-4f));
    printf("[PASS] riemannian_distance(C, C) = %.6f  (expected 0.0)\n", d);
}

static void test_riemannian_symmetry()
{
    // δ_R(C1, C2) = δ_R(C2, C1)
    Matrix C1 = {
        {4.0f, 0.5f},
        {0.5f, 2.0f}
    };
    Matrix C2 = {
        {2.0f, 0.3f},
        {0.3f, 5.0f}
    };
    float d12 = riemannian_distance(C1, C2);
    float d21 = riemannian_distance(C2, C1);
    assert(near(d12, d21, 1e-3f));
    printf("[PASS] riemannian_distance : symetrie d12=%.4f  d21=%.4f\n", d12, d21);
}

static void test_riemannian_identity_scale()
{
    // δ_R(I, α*I) = sqrt(N) * |ln(α)|  pour α > 0
    // Les val. propres de I^{-1/2} * αI * I^{-1/2} = αI sont toutes = α
    // → δ_R = sqrt(Σ ln²(α)) = sqrt(N) * |ln(α)|
    // N=2, α=4 → δ_R = sqrt(2) * ln(4) ≈ 1.9605
    Matrix I2 = identity(2);
    Matrix sI = {
        {4.0f, 0.0f},
        {0.0f, 4.0f}
    };
    float d = riemannian_distance(I2, sI);
    float expected = std::sqrt(2.0f) * std::log(4.0f);
    assert(near(d, expected, 1e-3f));
    printf("[PASS] riemannian_distance(I, 4I) = %.4f  (expected %.4f)\n", d, expected);
}

// ─── Tests mahalanobis_distance ───────────────────────────────────────────────

static void test_mahalanobis_identity_cov()
{
    // Σ^{-1} = I → D_M = distance Euclidienne
    std::vector<float> x = {3.0f, 4.0f};
    std::vector<float> mu = {0.0f, 0.0f};
    Matrix Sigma_inv = identity(2);
    float d = mahalanobis_distance(x, mu, Sigma_inv);
    assert(near(d, 5.0f)); // sqrt(9+16) = 5
    printf("[PASS] mahalanobis_distance(x=[3,4], mu=[0,0], Sigma=I) = %.4f  (expected 5.0)\n", d);
}

static void test_mahalanobis_self()
{
    // D_M(mu, mu, Sigma) = 0
    std::vector<float> mu = {1.0f, 2.0f, 3.0f};
    Matrix Sigma_inv = identity(3);
    float d = mahalanobis_distance(mu, mu, Sigma_inv);
    assert(near(d, 0.0f));
    printf("[PASS] mahalanobis_distance(x=mu) = %.6f  (expected 0.0)\n", d);
}

static void test_mahalanobis_scaled()
{
    // Σ^{-1} = diag(1/4, 1/9) → D_M([2,3], [0,0]) = sqrt((4/4)+(9/9)) = sqrt(2)
    std::vector<float> x = {2.0f, 3.0f};
    std::vector<float> mu = {0.0f, 0.0f};
    Matrix Sigma_inv = {
        {0.25f, 0.0f       },
        {0.0f,  1.0f / 9.0f}
    };
    float d = mahalanobis_distance(x, mu, Sigma_inv);
    assert(near(d, std::sqrt(2.0f)));
    printf("[PASS] mahalanobis_distance(diag(4,9)) = %.4f  (expected %.4f)\n", d, std::sqrt(2.0f));
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main()
{
    printf("=== RiemannianGeometry — tests unitaires ===\n\n");

    printf("-- jacobi_eigen --\n");
    test_jacobi_2x2_known();
    test_jacobi_identity();
    test_jacobi_diagonal();

    printf("\n-- compute_covariance --\n");
    test_covariance_2d_known();
    test_covariance_uncorrelated();

    printf("\n-- riemannian_distance --\n");
    test_riemannian_self_distance();
    test_riemannian_symmetry();
    test_riemannian_identity_scale();

    printf("\n-- mahalanobis_distance --\n");
    test_mahalanobis_identity_cov();
    test_mahalanobis_self();
    test_mahalanobis_scaled();

    printf("\n[OK] Tous les tests sont passes.\n");
    return 0;
}
