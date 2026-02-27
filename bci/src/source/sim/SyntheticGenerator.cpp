/**
 * @file SyntheticGenerator.cpp
 * @brief Implementation of the deterministic EEG signal generator.
 * @author MasterLaplace
 */

#include "lpl/bci/source/sim/SyntheticGenerator.hpp"

#include <chrono>
#include <cmath>
#include <numbers>

namespace lpl::bci::source {

SyntheticGenerator::SyntheticGenerator(std::uint64_t seed, std::size_t channelCount)
    : _profile{}
    , _sampleIndex(0)
    , _blinkRemaining(0)
    , _channelCount(channelCount)
{
    if (seed == 0) {
        seed = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    _rng.seed(seed);
    _noiseDist = std::normal_distribution<float>(0.0f, _profile.noiseAmplitudeUv);
    _blinkDist = std::uniform_real_distribution<float>(0.0f, 1.0f);
}

void SyntheticGenerator::setProfile(const SyntheticProfile &profile)
{
    _profile = profile;
    _noiseDist = std::normal_distribution<float>(0.0f, _profile.noiseAmplitudeUv);
}

const SyntheticProfile &SyntheticGenerator::profile() const noexcept
{
    return _profile;
}

std::vector<Sample> SyntheticGenerator::generate(std::size_t count)
{
    std::vector<Sample> output(count);

    for (std::size_t t = 0; t < count; ++t) {
        const float timeSec = static_cast<float>(_sampleIndex) / kDefaultSampleRate;

        if (_blinkRemaining <= 0 && _blinkDist(_rng) < _profile.blinkProbability) {
            _blinkRemaining = static_cast<int>(
                _profile.blinkDurationSec * kDefaultSampleRate);
        }

        float blinkEnvelope = 0.0f;
        if (_blinkRemaining > 0) {
            const float blinkSamples = _profile.blinkDurationSec * kDefaultSampleRate;
            const float progress =
                1.0f - static_cast<float>(_blinkRemaining) / blinkSamples;
            blinkEnvelope = _profile.blinkAmplitudeUv * 0.5f *
                (1.0f - std::cos(2.0f * std::numbers::pi_v<float> * progress));
            --_blinkRemaining;
        }

        output[t].channels.resize(_channelCount);
        output[t].timestamp = static_cast<double>(timeSec);

        for (std::size_t ch = 0; ch < _channelCount; ++ch) {
            float value = 0.0f;

            for (const auto &osc : _profile.oscillators) {
                const float phase =
                    2.0f * std::numbers::pi_v<float> * osc.freqHz * timeSec +
                    osc.phaseOffset +
                    static_cast<float>(ch) * _profile.channelPhaseSpread;
                value += osc.amplitudeUv * std::sin(phase);
            }

            value += _noiseDist(_rng);

            const float blinkWeight =
                (ch < 2) ? 1.0f : (ch < 4) ? 0.3f : 0.05f;
            value += blinkEnvelope * blinkWeight;

            output[t].channels[ch] = value;
        }

        ++_sampleIndex;
    }

    return output;
}

void SyntheticGenerator::reset(std::uint64_t seed)
{
    if (seed == 0) {
        seed = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    _rng.seed(seed);
    _sampleIndex = 0;
    _blinkRemaining = 0;
}

std::uint64_t SyntheticGenerator::sampleIndex() const noexcept
{
    return _sampleIndex;
}

} // namespace lpl::bci::source
