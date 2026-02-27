/**
 * @file Riemannian.hpp
 * @brief Riemannian geometry on the SPD manifold for BCI classification.
 * @author MasterLaplace
 *
 * Implements matrix logarithm, square-root, geodesic distance, Mahalanobis
 * distance, and Fréchet mean on the manifold of Symmetric Positive-Definite
 * (SPD) matrices. All operations use Eigen for eigen-decomposition.
 *
 * The SPD manifold equips covariance matrices with a natural metric that is
 * invariant to affine transformations, making these tools ideal for BCI
 * feature extraction and transfer learning.
 *
 * @see Covariance
 */

#pragma once

#include "lpl/bci/core/Error.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <span>
#include <vector>

namespace lpl::bci::math {

/**
 * @brief Computes the matrix square root of a symmetric positive-definite matrix.
 *
 * Uses eigenvalue decomposition: @f$ S^{1/2} = V \, \Lambda^{1/2} \, V^T @f$
 *
 * @param spd Symmetric positive-definite matrix
 * @return Matrix square root, or Error if matrix is not valid SPD
 */
[[nodiscard]] Expected<Eigen::MatrixXf> matrixSqrt(const Eigen::MatrixXf& spd);

/**
 * @brief Computes the inverse matrix square root of a SPD matrix.
 *
 * @f$ S^{-1/2} = V \, \Lambda^{-1/2} \, V^T @f$
 *
 * @param spd Symmetric positive-definite matrix
 * @return Inverse square root, or Error if matrix is not valid SPD
 */
[[nodiscard]] Expected<Eigen::MatrixXf> matrixSqrtInv(const Eigen::MatrixXf& spd);

/**
 * @brief Computes the matrix logarithm of a SPD matrix.
 *
 * @f$ \log(S) = V \, \log(\Lambda) \, V^T @f$
 *
 * @param spd Symmetric positive-definite matrix
 * @return Matrix logarithm, or Error if matrix is not valid SPD
 */
[[nodiscard]] Expected<Eigen::MatrixXf> matrixLog(const Eigen::MatrixXf& spd);

/**
 * @brief Computes the affine-invariant Riemannian distance between two SPD matrices.
 *
 * @f$ d(A, B) = \| \log(A^{-1/2} \, B \, A^{-1/2}) \|_F @f$
 *
 * @param a First SPD matrix
 * @param b Second SPD matrix
 * @return Geodesic distance, or Error on invalid input
 */
[[nodiscard]] Expected<float> riemannianDistance(
    const Eigen::MatrixXf& a,
    const Eigen::MatrixXf& b);

/**
 * @brief Computes the Mahalanobis distance between a vector and a distribution.
 *
 * @f$ d_M(x, \mu, \Sigma) = \sqrt{(x - \mu)^T \, \Sigma^{-1} \, (x - \mu)} @f$
 *
 * @param sample Observation vector
 * @param mean   Distribution mean
 * @param cov    Covariance matrix (must be invertible)
 * @return Mahalanobis distance, or Error on singular covariance
 */
[[nodiscard]] Expected<float> mahalanobisDistance(
    const Eigen::VectorXf& sample,
    const Eigen::VectorXf& mean,
    const Eigen::MatrixXf& cov);

/**
 * @brief Computes the Fréchet mean (geometric mean) of SPD matrices.
 *
 * Iterative fixed-point algorithm on the SPD manifold:
 *
 * @f$ M_{k+1} = M_k^{1/2} \exp\!\Bigl(\frac{1}{N} \sum_{i=1}^{N}
 *    \log\bigl(M_k^{-1/2} \, C_i \, M_k^{-1/2}\bigr)\Bigr) M_k^{1/2} @f$
 *
 * @param matrices   Span of SPD matrices (same dimensions)
 * @param maxIter    Maximum number of iterations (default: 50)
 * @param tolerance  Convergence threshold on the Frobenius norm (default: 1e-6)
 * @return Fréchet mean matrix, or Error on invalid input
 */
[[nodiscard]] Expected<Eigen::MatrixXf> frechetMean(
    std::span<const Eigen::MatrixXf> matrices,
    std::size_t maxIter = 50,
    float tolerance = 1e-6f);

} // namespace lpl::bci::math
