/**
 * @file StabilityMonitorBox.cpp
 * @brief Implementation of the EMA-smoothed stability monitor.
 */

#include "lpl/bci/openvibe/StabilityMonitorBox.hpp"

#include <algorithm>

namespace bci::openvibe {

StabilityMonitorBox::StabilityMonitorBox(const StabilityMonitorConfig& config)
    : _config(config)
    , _metric(metric::StabilityConfig{
          .historySize      = config.historySize,
          .stableThreshold  = config.stableThreshold,
          .unstableThreshold = config.maxExpectedDist,
      })
{
}

Expected<StabilityMonitorResult> StabilityMonitorBox::update(
    const Eigen::MatrixXf& covariance)
{
    auto result = _metric.update(covariance);
    if (!result)
        return std::unexpected(result.error());

    const float normalizedDist = std::min(
        result->currentDistance / _config.maxExpectedDist, 1.0f);
    const float instantStability = 1.0f - normalizedDist;

    _smoothedStability = _smoothedStability * (1.0f - _config.smoothingFactor)
                       + instantStability * _config.smoothingFactor;

    return StabilityMonitorResult{
        .rawDistance        = result->currentDistance,
        .smoothedStability  = _smoothedStability,
        .isStable           = (_smoothedStability > 0.6f),
    };
}

float StabilityMonitorBox::stability() const noexcept
{
    return _smoothedStability;
}

void StabilityMonitorBox::reset() noexcept
{
    _metric.reset();
    _smoothedStability = 0.5f;
}

} // namespace bci::openvibe
