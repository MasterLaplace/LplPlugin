/**
 * @file StabilityMetric.cpp
 * @brief Implementation of the temporal stability metric.
 *
 * Fixes V1's CBoxAlgorithmStabilityMonitor bug where _distanceHistory was
 * declared but never populated, causing the stability assessment to always
 * report default values.
 */

#include "metric/StabilityMetric.hpp"
#include "math/Riemannian.hpp"

#include <cmath>
#include <numeric>

namespace bci::metric {

StabilityMetric::StabilityMetric(const StabilityConfig& config)
    : _config(config)
{
}

Expected<StabilityResult> StabilityMetric::update(const Eigen::MatrixXf& covariance)
{
    if (!_hasPrevious) {
        _previousCov = covariance;
        _hasPrevious = true;
        return StabilityResult{
            .currentDistance = 0.0f,
            .meanDistance    = 0.0f,
            .rmsDistance     = 0.0f,
            .isStable        = true,
        };
    }

    auto distance = math::riemannianDistance(_previousCov, covariance);
    if (!distance)
        return std::unexpected(distance.error());

    _distanceHistory.push_back(*distance);

    while (_distanceHistory.size() > _config.historySize)
        _distanceHistory.pop_front();

    _previousCov = covariance;

    const auto n = static_cast<float>(_distanceHistory.size());

    const float meanDist = std::accumulate(
        _distanceHistory.begin(), _distanceHistory.end(), 0.0f) / n;

    float sumSq = 0.0f;
    for (const float d : _distanceHistory)
        sumSq += d * d;

    const float rmsDist = std::sqrt(sumSq / n);

    return StabilityResult{
        .currentDistance = *distance,
        .meanDistance    = meanDist,
        .rmsDistance     = rmsDist,
        .isStable        = (rmsDist < _config.stableThreshold),
    };
}

std::size_t StabilityMetric::historySize() const noexcept
{
    return _distanceHistory.size();
}

void StabilityMetric::reset() noexcept
{
    _distanceHistory.clear();
    _previousCov = Eigen::MatrixXf{};
    _hasPrevious = false;
}

} // namespace bci::metric
