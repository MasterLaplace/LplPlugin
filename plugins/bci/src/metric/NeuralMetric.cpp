/**
 * @file NeuralMetric.cpp
 * @brief Implementation of neural state normalization.
 */

#include "metric/NeuralMetric.hpp"

#include <algorithm>
#include <cmath>

namespace bci::metric {

NeuralMetric::NeuralMetric(const NeuralMetricConfig& config)
    : _config(config)
{
}

void NeuralMetric::setBaselines(std::span<const ChannelBaseline> baselines) noexcept
{
    _baselines.assign(baselines.begin(), baselines.end());
}

NeuralState NeuralMetric::compute(
    std::span<const float> alphaPowers,
    std::span<const float> betaPowers) const noexcept
{
    const std::size_t channelCount = std::min({
        alphaPowers.size(),
        betaPowers.size(),
        _baselines.size(),
    });

    NeuralState state;
    state.channelAlpha.resize(channelCount);
    state.channelBeta.resize(channelCount);

    float alphaSum = 0.0f;
    float betaSum  = 0.0f;

    for (std::size_t ch = 0; ch < channelCount; ++ch) {
        const auto& bl = _baselines[ch];

        state.channelAlpha[ch] = normalize(
            alphaPowers[ch], bl.alpha.mean, bl.alpha.stdDev, _config.kSigma);
        state.channelBeta[ch] = normalize(
            betaPowers[ch], bl.beta.mean, bl.beta.stdDev, _config.kSigma);

        alphaSum += state.channelAlpha[ch];
        betaSum  += state.channelBeta[ch];
    }

    if (channelCount > 0) {
        const auto n = static_cast<float>(channelCount);
        state.alphaPower = alphaSum / n;
        state.betaPower  = betaSum / n;
    }

    return state;
}

float NeuralMetric::normalize(
    float value,
    float mean,
    float stdDev,
    float kSigma) noexcept
{
    const float range = 2.0f * kSigma * stdDev;

    if (range <= 0.0f)
        return 0.5f;

    const float lower = mean - kSigma * stdDev;
    return std::clamp((value - lower) / range, 0.0f, 1.0f);
}

} // namespace bci::metric
