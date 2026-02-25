/**
 * @file BrainFlowSource.cpp
 * @brief Implementation of the BrainFlow acquisition source.
 * @author MasterLaplace
 */

#include "source/BrainFlowSource.hpp"
#include "core/Constants.hpp"

#include <algorithm>

namespace bci::source {

BrainFlowSource::BrainFlowSource(BrainFlowConfig config)
    : _config(std::move(config))
{
}

BrainFlowSource::~BrainFlowSource()
{
    stop();
}

#ifdef LPL_HAS_BRAINFLOW

ExpectedVoid BrainFlowSource::start()
{
    if (_running) {
        return std::unexpected(
            Error::make(ErrorCode::kAlreadyRunning, "BrainFlowSource already running"));
    }

    try {
        BrainFlowInputParams params;
        if (!_config.serialPort.empty()) {
            params.serial_port = _config.serialPort;
        }
        if (!_config.serialNumber.empty()) {
            params.serial_number = _config.serialNumber;
        }

        _board = std::make_unique<BoardShim>(_config.boardId, params);
        _board->prepare_session();
        _eegChannels = BoardShim::get_eeg_channels(_config.boardId);
        _sampleRate = BoardShim::get_sampling_rate(_config.boardId);
        _board->start_stream();
        _running = true;

        return {};
    } catch (const BrainFlowException &e) {
        return std::unexpected(
            Error::make(ErrorCode::kBrainFlowInitFailed,
                std::string(e.what()) + " (code: " + std::to_string(e.exit_code) + ")"));
    }
}

Expected<std::size_t> BrainFlowSource::read(std::span<Sample> buffer)
{
    if (!_running || !_board) {
        return std::unexpected(
            Error::make(ErrorCode::kNotInitialized, "BrainFlowSource not started"));
    }

    try {
        BrainFlowArray<double, 2> data = _board->get_board_data();
        const auto numSamples = static_cast<std::size_t>(data.get_size(1));

        if (numSamples == 0) {
            return std::size_t{0};
        }

        const std::size_t chCount = std::min(
            _eegChannels.size(), static_cast<std::size_t>(kDefaultChannelCount));
        const std::size_t count = std::min(numSamples, buffer.size());

        for (std::size_t s = 0; s < count; ++s) {
            buffer[s].channels.resize(chCount);
            for (std::size_t ch = 0; ch < chCount; ++ch) {
                buffer[s].channels[ch] =
                    static_cast<float>(data.at(_eegChannels[ch], static_cast<int>(s)));
            }
            buffer[s].timestamp = static_cast<double>(s) / static_cast<double>(_sampleRate);
        }

        return count;
    } catch (const BrainFlowException &e) {
        return std::unexpected(
            Error::make(ErrorCode::kBrainFlowStreamFailed, e.what()));
    }
}

void BrainFlowSource::stop() noexcept
{
    if (!_running || !_board) {
        return;
    }

    try {
        _board->stop_stream();
        _board->release_session();
    } catch (...) { }

    _running = false;
}

SourceInfo BrainFlowSource::info() const noexcept
{
    return SourceInfo{
        .name = "BrainFlow (board " + std::to_string(_config.boardId) + ")",
        .channelCount = _eegChannels.size(),
        .sampleRate = static_cast<float>(_sampleRate)
    };
}

#else // !LPL_HAS_BRAINFLOW

ExpectedVoid BrainFlowSource::start()
{
    return std::unexpected(
        Error::make(ErrorCode::kBrainFlowInitFailed,
            "BrainFlow support not compiled (LPL_HAS_BRAINFLOW not defined)"));
}

Expected<std::size_t> BrainFlowSource::read(std::span<Sample>)
{
    return std::unexpected(
        Error::make(ErrorCode::kNotInitialized, "BrainFlow not available"));
}

void BrainFlowSource::stop() noexcept { }

SourceInfo BrainFlowSource::info() const noexcept
{
    return SourceInfo{.name = "BrainFlow (unavailable)", .channelCount = 0, .sampleRate = 0.0f};
}

#endif // LPL_HAS_BRAINFLOW

} // namespace bci::source
