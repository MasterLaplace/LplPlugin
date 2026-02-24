/**
 * @file SignalMetric.cpp
 * @brief Implementation of the Schumacher spectral ratio metric.
 */

#include "lpl/bci/metric/SignalMetric.hpp"
#include "lpl/bci/math/Statistics.hpp"

#include <numeric>

namespace bci::metric {

SignalMetric::SignalMetric(const SignalMetricConfig& config)
    : _thetaLow(math::Statistics::hzToBin(config.theta.low, config.sampleRate, config.fftSize))
    , _thetaHigh(math::Statistics::hzToBin(config.theta.high, config.sampleRate, config.fftSize))
    , _alphaLow(math::Statistics::hzToBin(config.alpha.low, config.sampleRate, config.fftSize))
    , _alphaHigh(math::Statistics::hzToBin(config.alpha.high, config.sampleRate, config.fftSize))
    , _betaLow(math::Statistics::hzToBin(config.beta.low, config.sampleRate, config.fftSize))
    , _betaHigh(math::Statistics::hzToBin(config.beta.high, config.sampleRate, config.fftSize))
{
}

std::vector<SignalMetricResult> SignalMetric::compute(
    std::span<const std::vector<float>> psd) const noexcept
{
    std::vector<SignalMetricResult> results;
    results.reserve(psd.size());

    for (const auto& channelPsd : psd) {
        const std::span<const float> bins(channelPsd);

        const float theta = math::Statistics::integratePsd(bins, _thetaLow, _thetaHigh);
        const float alpha = math::Statistics::integratePsd(bins, _alphaLow, _alphaHigh);
        const float beta  = math::Statistics::integratePsd(bins, _betaLow, _betaHigh);

        const float denominator = alpha + theta;
        const float ratio = (denominator > 0.0f) ? (beta / denominator) : 0.0f;

        results.push_back({
            .thetaPower = theta,
            .alphaPower = alpha,
            .betaPower  = beta,
            .ratio      = ratio,
        });
    }

    return results;
}

float SignalMetric::computeMean(
    std::span<const std::vector<float>> psd) const noexcept
{
    const auto results = compute(psd);

    if (results.empty())
        return 0.0f;

    const float sum = std::accumulate(
        results.begin(), results.end(), 0.0f,
        [](float acc, const SignalMetricResult& r) { return acc + r.ratio; });

    return sum / static_cast<float>(results.size());
}

} // namespace bci::metric
