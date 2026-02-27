// File: RiemannianGeometryEigen.hpp
// Description: High-performance Riemannian geometry for BCI covariance matrices,
// backed by Eigen3 for production-grade matrix operations.
//
// Provides the same API as RiemannianGeometry.hpp but leverages Eigen's
// optimized BLAS/LAPACK backend for eigenvalue decomposition, matrix log/sqrt,
// and covariance computation.
//
// Compilation conditionnelle :
//   - LPL_USE_EIGEN : utilise Eigen3 (recommandé en production)
//   - Par défaut    : les fonctions sont des wrappers autour de RiemannianGeometry.hpp
//
// Références :
//   - Barachant et al. (2012) — "Multiclass Brain-Computer Interface Classification
//     by Riemannian Geometry"
//   - Eigen : https://eigen.tuxfamily.org/
//
// Auteur: MasterLaplace

#pragma once

#ifdef LPL_USE_EIGEN
// ═══════════════════════════════════════════════════════════════════════════════
//  EIGEN IMPLEMENTATION — optimized BLAS/LAPACK backend
// ═══════════════════════════════════════════════════════════════════════════════
#    include <Eigen/Dense>
#    include <cmath>
#    include <cstdio>
#    include <vector>

namespace RiemannianEigen {

using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;

/// Compute the covariance matrix of a multi-channel signal.
/// @param channels  Each inner vector is one channel's time-domain signal
/// @return          N×N covariance matrix (N = number of channels)
[[nodiscard]] inline MatrixXf compute_covariance(const std::vector<std::vector<float>> &channels)
{
    const int N = static_cast<int>(channels.size());
    const int T = static_cast<int>(channels[0].size());

    // Build data matrix (N channels × T samples)
    MatrixXf data(N, T);
    for (int ch = 0; ch < N; ++ch)
        for (int t = 0; t < T; ++t)
            data(ch, t) = channels[ch][t];

    // Center each channel (subtract mean)
    VectorXf means = data.rowwise().mean();
    data.colwise() -= means;

    // Covariance: (1/T) * X * X^T
    return (data * data.transpose()) / static_cast<float>(T);
}

/// Matrix square root of a symmetric positive definite (SPD) matrix.
/// Uses eigendecomposition: C^{1/2} = V * diag(sqrt(λ)) * V^T
[[nodiscard]] inline MatrixXf matrix_sqrt(const MatrixXf &C)
{
    Eigen::SelfAdjointEigenSolver<MatrixXf> solver(C);
    VectorXf eigenvalues = solver.eigenvalues().cwiseMax(1e-10f).cwiseSqrt();
    return solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
}

/// Matrix inverse square root: C^{-1/2} = V * diag(1/sqrt(λ)) * V^T
[[nodiscard]] inline MatrixXf matrix_sqrt_inv(const MatrixXf &C)
{
    Eigen::SelfAdjointEigenSolver<MatrixXf> solver(C);
    VectorXf inv_sqrt_eigenvalues =
        solver.eigenvalues().cwiseMax(1e-10f).cwiseSqrt().cwiseInverse();
    return solver.eigenvectors() * inv_sqrt_eigenvalues.asDiagonal() *
           solver.eigenvectors().transpose();
}

/// Matrix logarithm of an SPD matrix: log(C) = V * diag(log(λ)) * V^T
[[nodiscard]] inline MatrixXf matrix_log(const MatrixXf &C)
{
    Eigen::SelfAdjointEigenSolver<MatrixXf> solver(C);
    VectorXf log_eigenvalues = solver.eigenvalues().cwiseMax(1e-10f).array().log().matrix();
    return solver.eigenvectors() * log_eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
}

/// Affine-invariant Riemannian distance between two SPD matrices.
///
/// δ_R(C1, C2) = || log(C1^{-1/2} · C2 · C1^{-1/2}) ||_F
///
/// This is the gold standard distance metric for BCI covariance matrices
/// (Barachant et al., 2012). It is invariant to affine transformations,
/// making it robust to electrode impedance changes and inter-session variability.
///
/// @param C1  First SPD covariance matrix
/// @param C2  Second SPD covariance matrix
/// @return    Riemannian distance (scalar ≥ 0)
[[nodiscard]] inline float riemannian_distance(const MatrixXf &C1, const MatrixXf &C2)
{
    MatrixXf C1_inv_sqrt = matrix_sqrt_inv(C1);
    MatrixXf M = C1_inv_sqrt * C2 * C1_inv_sqrt;

    Eigen::SelfAdjointEigenSolver<MatrixXf> solver(M);
    VectorXf log_eigenvalues = solver.eigenvalues().cwiseMax(1e-10f).array().log().matrix();

    return log_eigenvalues.norm();
}

/// Mahalanobis distance between a test point (covariance matrix) and a
/// reference distribution characterized by a mean covariance and its inverse.
///
/// D_M = sqrt(vec(C - μ)^T · Σ^{-1} · vec(C - μ))
///
/// Simplified: uses Frobenius norm of the difference projected through
/// the inverse reference covariance.
///
/// @param C       Test covariance matrix
/// @param C_ref   Reference (mean) covariance matrix
/// @return        Mahalanobis distance (scalar ≥ 0)
[[nodiscard]] inline float mahalanobis_distance(const MatrixXf &C, const MatrixXf &C_ref)
{
    MatrixXf C_ref_inv = C_ref.inverse();
    MatrixXf diff = C - C_ref;
    MatrixXf projected = C_ref_inv * diff;
    return std::sqrt(projected.squaredNorm());
}

/// Compute the Riemannian mean (Fréchet/Karcher mean) of a set of SPD matrices.
///
/// Iterative algorithm:
///   1. Start with arithmetic mean
///   2. Project to tangent space, average, project back
///   3. Repeat until convergence
///
/// @param matrices  Vector of SPD covariance matrices
/// @param maxIter   Maximum iterations (default: 50)
/// @param tol       Convergence tolerance (default: 1e-6)
/// @return          Riemannian mean matrix
[[nodiscard]] inline MatrixXf riemannian_mean(const std::vector<MatrixXf> &matrices, int maxIter = 50,
                                               float tol = 1e-6f)
{
    if (matrices.empty())
        return MatrixXf();

    const int N = static_cast<int>(matrices.size());
    MatrixXf mean = matrices[0]; // Initialize with first matrix

    for (int iter = 0; iter < maxIter; ++iter)
    {
        MatrixXf mean_sqrt_inv = matrix_sqrt_inv(mean);
        MatrixXf S = MatrixXf::Zero(mean.rows(), mean.cols());

        for (const auto &Ci : matrices)
        {
            MatrixXf M = mean_sqrt_inv * Ci * mean_sqrt_inv;
            S += matrix_log(M);
        }

        S /= static_cast<float>(N);

        if (S.norm() < tol)
            break;

        // Update mean: mean = mean^{1/2} · exp(S) · mean^{1/2}
        MatrixXf mean_sqrt = matrix_sqrt(mean);

        // exp(S) via eigendecomposition
        Eigen::SelfAdjointEigenSolver<MatrixXf> solver(S);
        VectorXf exp_eigenvalues = solver.eigenvalues().array().exp().matrix();
        MatrixXf expS =
            solver.eigenvectors() * exp_eigenvalues.asDiagonal() * solver.eigenvectors().transpose();

        mean = mean_sqrt * expS * mean_sqrt;
    }

    return mean;
}

} // namespace RiemannianEigen

#else
// ═══════════════════════════════════════════════════════════════════════════════
//  FALLBACK — includes the hand-rolled implementation
// ═══════════════════════════════════════════════════════════════════════════════
#    include "RiemannianGeometry.hpp"

// When Eigen is not available, the original RiemannianGeometry namespace
// provides all necessary functions (riemannian_distance, mahalanobis_distance,
// compute_covariance) using manual Jacobi eigenvalue decomposition.

#endif // LPL_USE_EIGEN
