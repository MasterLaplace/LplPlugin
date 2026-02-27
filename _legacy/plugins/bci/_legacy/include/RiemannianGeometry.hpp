// File: RiemannianGeometry.hpp
// Description: Géométrie Riemannienne pour matrices SPD + distance de Mahalanobis
// Références:
//   Moakher 2005 — A Differential Geometric Approach to the Geometric Mean of SPD matrices
//   Arsigny et al. 2006 — Log-Euclidean metrics for fast and simple SPD matrices
//   Blankertz et al. 2011 — Single-trial EEG using Riemannian covariance matrices
// Auteur: MasterLaplace

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace RiemannianGeometry {

using Matrix = std::vector<std::vector<float>>;

// ─── Utilitaires matriciels ───────────────────────────────────────────────────

/// Crée une matrice N×N de zéros.
inline Matrix zeros(size_t n) { return Matrix(n, std::vector<float>(n, 0.0f)); }

/// Crée une matrice identité N×N.
inline Matrix identity(size_t n)
{
    Matrix M = zeros(n);
    for (size_t i = 0u; i < n; ++i)
        M[i][i] = 1.0f;
    return M;
}

/// Produit matriciel C = A × B (N×N).
inline Matrix matmul(const Matrix &A, const Matrix &B)
{
    const size_t n = A.size();
    Matrix C = zeros(n);
    for (size_t i = 0u; i < n; ++i)
        for (size_t k = 0u; k < n; ++k)
            for (size_t j = 0u; j < n; ++j)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

/// Transposée d'une matrice N×N.
inline Matrix transpose(const Matrix &A)
{
    const size_t n = A.size();
    Matrix T = zeros(n);
    for (size_t i = 0u; i < n; ++i)
        for (size_t j = 0u; j < n; ++j)
            T[i][j] = A[j][i];
    return T;
}

// ─── Décomposition propre (Jacobi) ───────────────────────────────────────────

/// Décomposition propre d'une matrice symétrique réelle par l'algorithme de Jacobi.
///
/// Après appel :
///   - M[i][i]       ≈ λ_i  (matrice diagonalisée in-place)
///   - eigenvalues   = {λ_0, ..., λ_{N-1}}
///   - V             = matrice des vecteurs propres (colonnes)
///
/// Complexité : O(N³) par sweep, convergence typique en 5–15 sweeps.
inline void jacobi_eigen(Matrix &M, std::vector<float> &eigenvalues, Matrix &V)
{
    const size_t n = M.size();
    V = identity(n);
    eigenvalues.resize(n);

    constexpr int MAX_SWEEPS = 150;
    constexpr float TOL = 1e-8f;

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep)
    {
        float off_diag = 0.0f;
        for (size_t i = 0u; i < n; ++i)
            for (size_t j = i + 1u; j < n; ++j)
                off_diag += M[i][j] * M[i][j];
        if (off_diag < TOL)
            break;

        for (size_t p = 0u; p < n - 1u; ++p)
        {
            for (size_t q = p + 1u; q < n; ++q)
            {
                if (std::abs(M[p][q]) < 1e-12f)
                    continue;

                // Rotation de Givens optimale (algorithme de Rutishauser)
                const float theta = 0.5f * (M[q][q] - M[p][p]) / M[p][q];
                const float t = (theta >= 0.0f) ? 1.0f / (theta + std::sqrt(1.0f + theta * theta)) :
                                                  1.0f / (theta - std::sqrt(1.0f + theta * theta));
                const float c = 1.0f / std::sqrt(1.0f + t * t);
                const float s = t * c;
                const float tau = s / (1.0f + c);

                const float d_pp = M[p][p] - t * M[p][q];
                const float d_qq = M[q][q] + t * M[p][q];
                M[p][p] = d_pp;
                M[q][q] = d_qq;
                M[p][q] = M[q][p] = 0.0f;

                for (size_t r = 0u; r < n; ++r)
                {
                    if (r == p || r == q)
                        continue;
                    const float g = M[r][p];
                    const float h = M[r][q];
                    M[r][p] = M[p][r] = g - s * (h + g * tau);
                    M[r][q] = M[q][r] = h + s * (g - h * tau);
                }

                for (size_t r = 0u; r < n; ++r)
                {
                    const float g = V[r][p];
                    const float h = V[r][q];
                    V[r][p] = g - s * (h + g * tau);
                    V[r][q] = h + s * (g - h * tau);
                }
            }
        }
    }

    for (size_t i = 0u; i < n; ++i)
        eigenvalues[i] = M[i][i];
}

// ─── Racine carrée et inverse de matrice SPD ──────────────────────────────────

/// Calcule M^{+1/2} d'une matrice SPD : M^{1/2} = V * diag(sqrt(|λ|)) * V^T
inline Matrix matrix_sqrt(const Matrix &M)
{
    Matrix tmp = M;
    std::vector<float> eigenvalues;
    Matrix V;
    jacobi_eigen(tmp, eigenvalues, V);

    const size_t n = M.size();
    Matrix S = zeros(n);
    for (size_t i = 0u; i < n; ++i)
        for (size_t j = 0u; j < n; ++j)
            for (size_t k = 0u; k < n; ++k)
                S[i][j] += V[i][k] * std::sqrt(std::max(0.0f, eigenvalues[k])) * V[j][k];
    return S;
}

/// Calcule M^{-1/2} d'une matrice SPD : M^{-1/2} = V * diag(1/sqrt(max(λ,ε))) * V^T
inline Matrix matrix_sqrt_inv(const Matrix &M, float eps = 1e-8f)
{
    Matrix tmp = M;
    std::vector<float> eigenvalues;
    Matrix V;
    jacobi_eigen(tmp, eigenvalues, V);

    const size_t n = M.size();
    Matrix S = zeros(n);
    for (size_t i = 0u; i < n; ++i)
        for (size_t j = 0u; j < n; ++j)
            for (size_t k = 0u; k < n; ++k)
            {
                const float lam = std::max(eigenvalues[k], eps);
                S[i][j] += V[i][k] * (1.0f / std::sqrt(lam)) * V[j][k];
            }
    return S;
}

/// Calcule M^{-1} d'une matrice SPD : M^{-1} = V * diag(1/max(λ,ε)) * V^T
inline Matrix matrix_inv(const Matrix &M, float eps = 1e-8f)
{
    Matrix tmp = M;
    std::vector<float> eigenvalues;
    Matrix V;
    jacobi_eigen(tmp, eigenvalues, V);

    const size_t n = M.size();
    Matrix S = zeros(n);
    for (size_t i = 0u; i < n; ++i)
        for (size_t j = 0u; j < n; ++j)
            for (size_t k = 0u; k < n; ++k)
            {
                const float lam = std::max(eigenvalues[k], eps);
                S[i][j] += V[i][k] * (1.0f / lam) * V[j][k];
            }
    return S;
}

// ─── Matrice de covariance ────────────────────────────────────────────────────

/// Calcule la matrice de covariance empirique N×N d'une fenêtre temporelle.
/// window[t][ch] : t = indice temporel, ch = canal (0..N-1)
/// Retourne une matrice SPD de taille N×N (covariance corrigée Bessel).
inline Matrix compute_covariance(const std::vector<std::vector<float>> &window) noexcept
{
    if (window.empty())
        return {};
    const size_t T = window.size();
    const size_t N = window[0].size();

    std::vector<float> mean(N, 0.0f);
    for (const auto &sample : window)
        for (size_t i = 0u; i < N; ++i)
            mean[i] += sample[i];
    for (float &m : mean)
        m /= static_cast<float>(T);

    Matrix C = zeros(N);
    for (const auto &sample : window)
    {
        for (size_t i = 0u; i < N; ++i)
            for (size_t j = i; j < N; ++j)
            {
                const float v = (sample[i] - mean[i]) * (sample[j] - mean[j]);
                C[i][j] += v;
                if (i != j)
                    C[j][i] += v;
            }
    }

    const float scale = 1.0f / static_cast<float>(T > 1u ? T - 1u : 1u);
    for (auto &row : C)
        for (float &v : row)
            v *= scale;

    return C;
}

// ─── Distance de Riemannian ───────────────────────────────────────────────────

/// Distance géodésique (Riemannienne affine-invariante) entre deux matrices SPD.
///
///   δ_R(C1, C2) = ‖ log(C1^{-1/2} C2 C1^{-1/2}) ‖_F
///               = sqrt( Σ_i ln²(λ_i) )
///
/// où λ_i sont les valeurs propres de M = C1^{-1/2} C2 C1^{-1/2}.
///
/// Propriété : δ_R(C, C) = 0,  δ_R(C1, C2) = δ_R(C2, C1).
[[nodiscard]] inline float riemannian_distance(const Matrix &C1, const Matrix &C2)
{
    const Matrix C1_inv_sqrt = matrix_sqrt_inv(C1);
    Matrix M = matmul(C1_inv_sqrt, matmul(C2, C1_inv_sqrt));

    std::vector<float> eigenvalues;
    Matrix V;
    jacobi_eigen(M, eigenvalues, V);

    constexpr float EPS = 1e-10f;
    float sum_ln2 = 0.0f;
    for (const float lam : eigenvalues)
    {
        const float safe = std::max(lam, EPS);
        const float ln = std::log(safe);
        sum_ln2 += ln * ln;
    }
    return std::sqrt(sum_ln2);
}

// ─── Distance de Mahalanobis ─────────────────────────────────────────────────

/// Distance de Mahalanobis du point x_t par rapport au centroïde μ_c.
///
///   D_M(x_t) = sqrt( (x_t - μ_c)^T * Σ_c^{-1} * (x_t - μ_c) )
///
/// Utilisé pour la détection d'anomalie dans l'espace des caractéristiques EEG.
/// Quand Σ_c^{-1} = I, se réduit à la distance Euclidienne.
///
/// @param x_t          Vecteur d'observation (taille N)
/// @param mu_c         Centroïde de la classe (taille N)
/// @param sigma_c_inv  Inverse de la covariance (N×N) — via matrix_inv(compute_covariance(...))
[[nodiscard]] inline float mahalanobis_distance(const std::vector<float> &x_t, const std::vector<float> &mu_c,
                                                const Matrix &sigma_c_inv) noexcept
{
    const size_t n = x_t.size();

    std::vector<float> diff(n);
    for (size_t i = 0u; i < n; ++i)
        diff[i] = x_t[i] - mu_c[i];

    std::vector<float> tmp(n, 0.0f);
    for (size_t i = 0u; i < n; ++i)
        for (size_t j = 0u; j < n; ++j)
            tmp[i] += sigma_c_inv[i][j] * diff[j];

    float dot = 0.0f;
    for (size_t i = 0u; i < n; ++i)
        dot += diff[i] * tmp[i];

    return std::sqrt(std::max(0.0f, dot));
}

} // namespace RiemannianGeometry
