/**
 * @file Covariance.hpp
 * @brief Covariance matrix estimation for multi-channel EEG data.
 * @author MasterLaplace
 *
 * Provides both batch (sample covariance) and online (Welford) estimators.
 * Produces Eigen MatrixXf results consumed by Riemannian geometry routines.
 *
 * @see Riemannian
 */

#pragma once

#include "core/Types.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <span>
#include <vector>

namespace bci::math {

/**
 * @brief Batch covariance matrix estimation from a sample matrix.
 *
 * Computes the unbiased sample covariance C = (1 / (N-1)) * X_c^T * X_c
 * where X_c is the column-centered data matrix.
 *
 * @param data  Sample matrix [samples x channels]
 * @return Covariance matrix [channels x channels], or zero matrix if < 2 samples
 */
[[nodiscard]] Eigen::MatrixXf batchCovariance(const Eigen::MatrixXf& data) noexcept;

/**
 * @brief Computes a regularized covariance matrix.
 *
 * @f$ C_{reg} = (1 - \alpha) \cdot C + \alpha \cdot \frac{\mathrm{tr}(C)}{p} \cdot I @f$
 *
 * Ledoit-Wolf shrinkage towards scaled identity for numerical stability.
 *
 * @param cov   Raw covariance matrix [p x p]
 * @param alpha Shrinkage parameter in [0, 1]
 * @return Regularized covariance matrix
 */
[[nodiscard]] Eigen::MatrixXf regularizeCovariance(
    const Eigen::MatrixXf& cov,
    float alpha = 0.01f) noexcept;

/**
 * @brief Online (Welford) covariance estimator.
 *
 * Incrementally updates mean and cross-product matrix with each new sample.
 * Memory-efficient for streaming BCI use-cases.
 */
class WelfordCovariance {
public:
    /**
     * @brief Constructs the estimator for a given number of channels.
     *
     * @param channelCount Number of channels
     */
    explicit WelfordCovariance(std::size_t channelCount);

    /**
     * @brief Feeds a single multi-channel sample into the estimator.
     *
     * @param sample One sample with channelCount values
     */
    void update(std::span<const float> sample) noexcept;

    /**
     * @brief Returns the current unbiased covariance estimate.
     *
     * @return Covariance [channels x channels], or zero matrix if < 2 updates
     */
    [[nodiscard]] Eigen::MatrixXf covariance() const noexcept;

    /**
     * @brief Returns the current sample mean per channel.
     */
    [[nodiscard]] const Eigen::VectorXf& mean() const noexcept;

    /**
     * @brief Number of samples fed so far.
     */
    [[nodiscard]] std::size_t count() const noexcept;

    /**
     * @brief Resets the estimator to its initial state.
     */
    void reset() noexcept;

private:
    std::size_t _channelCount;
    std::size_t _n = 0;
    Eigen::VectorXf _mean;
    Eigen::MatrixXf _coProduct;
};

} // namespace bci::math
