/**
 * @file StabilityMonitorBox.hpp
 * @brief OpenViBE-compatible EEG signal stability monitor.
 * @author MasterLaplace
 *
 * Implements Sollfrank et al. (2016) approach: monitors temporal stability
 * of EEG signal characteristics by computing the Riemannian distance between
 * consecutive covariance matrices. High distance = unstable mental state.
 *
 * Unlike the V1 implementation, this version properly populates the distance
 * history and delegates the Riemannian computation to StabilityMetric.
 *
 * OpenViBE integration:
 *   - Input 0  : Signal (EEG channels)
 *   - Output 0 : Streamed Matrix (1x1 — stability scalar [0, 1])
 *   - Setting 0: Smoothing factor (default: 0.1)
 *   - Setting 1: Max expected distance (default: 5.0)
 *   - Setting 2: History size (default: 30)
 *
 * @see StabilityMetric, Riemannian
 */

#pragma once

#include "lpl/bci/core/Error.hpp"
#include "lpl/bci/metric/StabilityMetric.hpp"

#include <Eigen/Dense>
#include <cstddef>

namespace lpl::bci::openvibe {

/**
 * @brief Configuration for the stability monitor box.
 */
struct StabilityMonitorConfig {
    float smoothingFactor   = 0.1f;
    float maxExpectedDist   = 5.0f;
    std::size_t historySize = 30;
    float stableThreshold   = 0.3f;
};

/**
 * @brief Smoothed stability result.
 */
struct StabilityMonitorResult {
    float rawDistance        = 0.0f;
    float smoothedStability = 0.5f;
    bool isStable           = false;
};

/**
 * @brief EMA-smoothed stability processor for OpenViBE integration.
 *
 * Wraps StabilityMetric with exponential moving average smoothing and
 * distance normalization for real-time display.
 */
class StabilityMonitorBox {
public:
    /**
     * @brief Constructs the monitor with the given configuration.
     *
     * @param config Smoothing and normalization parameters
     */
    explicit StabilityMonitorBox(const StabilityMonitorConfig& config = {});

    /**
     * @brief Feeds a new covariance matrix and returns smoothed stability.
     *
     * @param covariance Current trial's covariance matrix (SPD)
     * @return Smoothed stability result, or Error on Riemannian computation failure
     */
    [[nodiscard]] Expected<StabilityMonitorResult> update(
        const Eigen::MatrixXf& covariance);

    /**
     * @brief Returns the last smoothed stability value.
     */
    [[nodiscard]] float stability() const noexcept;

    /**
     * @brief Resets the monitor to its initial state.
     */
    void reset() noexcept;

private:
    StabilityMonitorConfig _config;
    metric::StabilityMetric _metric;
    float _smoothedStability = 0.5f;
};

} // namespace lpl::bci::openvibe
