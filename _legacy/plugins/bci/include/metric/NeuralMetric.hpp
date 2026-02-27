/**
 * @file NeuralMetric.hpp
 * @brief Transforms raw band powers into a normalized NeuralState.
 * @author MasterLaplace
 *
 * Maps per-channel alpha and beta band powers into the [0, 1] range using
 * baseline statistics (mean ± k·σ clamping). This provides a
 * hardware-independent representation of the user's neural state.
 *
 * @see Calibration, SignalMetric
 */

#pragma once

#include "core/Types.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace bci::metric {

/**
 * @brief Configuration for neural state normalization.
 */
struct NeuralMetricConfig {
    float kSigma = 2.0f;
};

/**
 * @brief Per-channel baseline statistics for normalization.
 */
struct ChannelBaseline {
    Baseline alpha;
    Baseline beta;
};

/**
 * @brief Normalizes raw band powers into a NeuralState using calibration baselines.
 *
 * The normalization formula for each value is:
 * @f$ v_{norm} = \text{clamp}\!\left(\frac{v - (\mu - k\sigma)}{2 k \sigma}, 0, 1\right) @f$
 */
class NeuralMetric {
public:
    /**
     * @brief Constructs the metric with the given configuration.
     *
     * @param config Normalization parameters
     */
    explicit NeuralMetric(const NeuralMetricConfig& config = {});

    /**
     * @brief Sets the per-channel baselines used for normalization.
     *
     * Must be called with calibration data before `compute()` produces
     * meaningful results.
     *
     * @param baselines Per-channel baseline statistics
     */
    void setBaselines(std::span<const ChannelBaseline> baselines) noexcept;

    /**
     * @brief Computes the normalized neural state from raw band powers.
     *
     * @param alphaPowers Per-channel alpha power values
     * @param betaPowers  Per-channel beta power values
     * @return NeuralState with normalized per-channel values in [0, 1]
     */
    [[nodiscard]] NeuralState compute(
        std::span<const float> alphaPowers,
        std::span<const float> betaPowers) const noexcept;

private:
    [[nodiscard]] static float normalize(
        float value,
        float mean,
        float stdDev,
        float kSigma) noexcept;

    NeuralMetricConfig _config;
    std::vector<ChannelBaseline> _baselines;
};

} // namespace bci::metric
