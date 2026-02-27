/**
 * @file StabilityMetric.hpp
 * @brief Temporal stability analysis of neural states on the SPD manifold.
 * @author MasterLaplace
 *
 * Evaluates how stable the user's neural state remains over a sliding window
 * by tracking Riemannian distances between successive covariance matrices.
 * A low stability score indicates consistent brain activity, which is
 * desirable for neurofeedback training.
 *
 * Fixes the V1 bug where _distanceHistory was never populated.
 *
 * @see Riemannian, NeuralMetric
 */

#pragma once

#include "core/Error.hpp"
#include "core/Types.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <deque>

namespace bci::metric {

/**
 * @brief Configuration for stability monitoring.
 */
struct StabilityConfig {
    std::size_t historySize = 30;
    float stableThreshold   = 0.3f;
    float unstableThreshold = 0.7f;
};

/**
 * @brief Stability assessment result.
 */
struct StabilityResult {
    float currentDistance = 0.0f;
    float meanDistance    = 0.0f;
    float rmsDistance     = 0.0f;
    bool isStable        = false;
};

/**
 * @brief Monitors temporal stability via Riemannian distances between successive
 *        covariance matrices.
 *
 * Maintains a sliding window of geodesic distances and provides aggregate
 * statistics for stability assessment.
 */
class StabilityMetric {
public:
    /**
     * @brief Constructs the metric with the given configuration.
     *
     * @param config Stability parameters
     */
    explicit StabilityMetric(const StabilityConfig& config = {});

    /**
     * @brief Feeds a new covariance matrix and returns the stability assessment.
     *
     * Computes the Riemannian distance to the previous matrix, appends it to the
     * sliding history, and produces aggregate metrics over the window.
     *
     * @param covariance Current trial's covariance matrix (SPD)
     * @return Stability result, or Error if Riemannian computation fails
     */
    [[nodiscard]] Expected<StabilityResult> update(const Eigen::MatrixXf& covariance);

    /**
     * @brief Returns the number of distance samples in the history.
     */
    [[nodiscard]] std::size_t historySize() const noexcept;

    /**
     * @brief Resets the history and previous covariance.
     */
    void reset() noexcept;

private:
    StabilityConfig _config;
    std::deque<float> _distanceHistory;
    Eigen::MatrixXf _previousCov;
    bool _hasPrevious = false;
};

} // namespace bci::metric
