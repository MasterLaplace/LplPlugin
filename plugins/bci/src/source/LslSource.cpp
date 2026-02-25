/**
 * @file LslSource.cpp
 * @brief Implementation of the LSL inlet acquisition source.
 * @author MasterLaplace
 */

#include "source/LslSource.hpp"

namespace bci::source {

LslSource::LslSource(LslSourceConfig config)
    : _config(std::move(config))
{
}

LslSource::~LslSource()
{
    stop();
}

ExpectedVoid LslSource::start()
{
    if (_running) {
        return std::unexpected(
            Error::make(ErrorCode::kAlreadyRunning, "LslSource already running"));
    }

    try {
        auto results = lsl::resolve_stream(
            "name", _config.streamName, 1, _config.resolveTimeoutSec);

        if (results.empty()) {
            return std::unexpected(
                Error::make(ErrorCode::kLslStreamNotFound,
                    "No LSL stream named '" + _config.streamName + "' found"));
        }

        _inlet = std::make_unique<lsl::stream_inlet>(results[0]);
        _channelCount = static_cast<std::size_t>(_inlet->info().channel_count());
        _sampleRate = _inlet->info().nominal_srate();
        _running = true;

        return {};
    } catch (const std::exception &e) {
        return std::unexpected(
            Error::make(ErrorCode::kLslConnectionFailed, e.what()));
    }
}

Expected<std::size_t> LslSource::read(std::span<Sample> buffer)
{
    if (!_running || !_inlet) {
        return std::unexpected(
            Error::make(ErrorCode::kNotInitialized, "LslSource not started"));
    }

    std::vector<float> raw(_channelCount);
    std::size_t count = 0;

    for (auto &sample : buffer) {
        double timestamp = _inlet->pull_sample(
            raw.data(), static_cast<int>(_channelCount), 0.0);

        if (timestamp == 0.0) {
            break;
        }

        sample.channels = raw;
        sample.timestamp = timestamp;
        ++count;
    }

    return count;
}

void LslSource::stop() noexcept
{
    _running = false;
    _inlet.reset();
}

SourceInfo LslSource::info() const noexcept
{
    return SourceInfo{
        .name = "LSL (" + _config.streamName + ")",
        .channelCount = _channelCount,
        .sampleRate = static_cast<float>(_sampleRate)
    };
}

} // namespace bci::source
