/**
 * @file MuscleRelaxationBox.cpp
 * @brief Implementation of the muscle relaxation processor.
 */

#include "lpl/bci/openvibe/MuscleRelaxationBox.hpp"
#include "lpl/bci/math/Statistics.hpp"

namespace bci::openvibe {

MuscleRelaxationBox::MuscleRelaxationBox(const MuscleRelaxationConfig& config)
    : _config(config)
    , _gammaLow(math::Statistics::hzToBin(config.lowerFreqHz, config.sampleRate, config.fftSize))
    , _gammaHigh(math::Statistics::hzToBin(config.upperFreqHz, config.sampleRate, config.fftSize))
{
}

MuscleRelaxationResult MuscleRelaxationBox::compute(
    std::span<const std::vector<float>> psd) const noexcept
{
    if (psd.empty())
        return {};

    float totalGamma = 0.0f;

    for (const auto& channelPsd : psd) {
        totalGamma += math::Statistics::integratePsd(
            channelPsd, _gammaLow, _gammaHigh);
    }

    const float meanGamma = totalGamma / static_cast<float>(psd.size());

    return {
        .gammaRatio = meanGamma,
        .isAlert    = (meanGamma > _config.alertThreshold),
    };
}

const MuscleRelaxationConfig& MuscleRelaxationBox::config() const noexcept
{
    return _config;
}

} // namespace bci::openvibe
