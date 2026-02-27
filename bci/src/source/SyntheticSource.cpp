/**
 * @file SyntheticSource.cpp
 * @brief Implementation of the synthetic EEG acquisition source.
 * @author MasterLaplace
 */

#include "lpl/bci/source/SyntheticSource.hpp"

#include <algorithm>

namespace lpl::bci::source {

SyntheticSource::SyntheticSource(
    std::uint64_t seed, bool realtime, std::size_t channelCount, float sampleRate)
    : _gen(seed, channelCount)
    , _realtime(realtime)
    , _sampleRate(sampleRate)
    , _channelCount(channelCount)
{
}

ExpectedVoid SyntheticSource::start()
{
    if (_running) {
        return std::unexpected(
            Error::make(ErrorCode::kAlreadyRunning, "SyntheticSource already running"));
    }

    _running = true;
    _lastRead = std::chrono::steady_clock::now();

    return {};
}

Expected<std::size_t> SyntheticSource::read(std::span<Sample> buffer)
{
    if (!_running) {
        return std::unexpected(
            Error::make(ErrorCode::kNotInitialized, "SyntheticSource not started"));
    }

    if (buffer.empty()) {
        return std::size_t{0};
    }

    std::size_t samplesToGenerate;
    if (_realtime) {
        auto now = std::chrono::steady_clock::now();
        const float elapsed =
            std::chrono::duration<float>(now - _lastRead).count();
        _lastRead = now;

        samplesToGenerate = std::max(
            std::size_t{1},
            static_cast<std::size_t>(elapsed * _sampleRate));
        samplesToGenerate = std::min(samplesToGenerate, kMaxSamplesPerUpdate);
    } else {
        samplesToGenerate = kFftUpdateInterval;
    }

    samplesToGenerate = std::min(samplesToGenerate, buffer.size());

    auto generated = _gen.generate(samplesToGenerate);
    for (std::size_t i = 0; i < samplesToGenerate; ++i) {
        buffer[i] = std::move(generated[i]);
    }

    return samplesToGenerate;
}

void SyntheticSource::stop() noexcept
{
    _running = false;
}

SourceInfo SyntheticSource::info() const noexcept
{
    return SourceInfo{
        .name = "Synthetic EEG",
        .channelCount = _channelCount,
        .sampleRate = _sampleRate
    };
}

SyntheticGenerator &SyntheticSource::generator() noexcept
{
    return _gen;
}

} // namespace lpl::bci::source
