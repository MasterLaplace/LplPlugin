/**
 * @file OpenBciSource.cpp
 * @brief Implementation of the OpenBCI Cyton serial acquisition source.
 * @author MasterLaplace
 */

#include "source/OpenBciSource.hpp"

#include <array>
#include <cstring>

namespace bci::source {

OpenBciSource::OpenBciSource(OpenBciConfig config)
    : _config(std::move(config))
{
}

OpenBciSource::~OpenBciSource()
{
    stop();
}

ExpectedVoid OpenBciSource::start()
{
    if (_started) {
        return std::unexpected(
            Error::make(ErrorCode::kAlreadyRunning, "OpenBciSource already started"));
    }

    SerialConfig serialCfg{
        .portPath = _config.port,
        .baudRate = _config.baudRate,
        .dataBits = 8,
        .stopBits = 1,
        .parity = false,
        .vmin = static_cast<std::uint8_t>(kCytonPacketSize),
        .vtime = 1
    };

    auto result = _serial.open(serialCfg);
    if (!result) {
        return std::unexpected(result.error());
    }

    _started = true;
    _worker = std::jthread([this](std::stop_token st) { workerLoop(st); });

    return {};
}

Expected<std::size_t> OpenBciSource::read(std::span<Sample> buffer)
{
    if (!_started) {
        return std::unexpected(
            Error::make(ErrorCode::kNotInitialized, "OpenBciSource not started"));
    }

    std::size_t count = 0;
    for (auto &sample : buffer) {
        if (!_ring.pop(sample)) {
            break;
        }
        ++count;
    }

    return count;
}

void OpenBciSource::stop() noexcept
{
    if (!_started) {
        return;
    }

    _worker.request_stop();
    _serial.close();

    if (_worker.joinable()) {
        _worker.join();
    }

    _started = false;
}

SourceInfo OpenBciSource::info() const noexcept
{
    return SourceInfo{
        .name = "OpenBCI Cyton (" + _config.port + ")",
        .channelCount = _config.channelCount,
        .sampleRate = kDefaultSampleRate
    };
}

void OpenBciSource::workerLoop(std::stop_token stopToken)
{
    std::array<std::uint8_t, kCytonPacketSize> buffer{};

    while (!stopToken.stop_requested()) {
        auto result = _serial.read(buffer);
        if (!result || result.value() != kCytonPacketSize) {
            if (stopToken.stop_requested())
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (buffer[0] != kCytonHeaderByte ||
            buffer[kCytonPacketSize - 1] != kCytonFooterByte) {

            std::uint8_t byte = 0;
            while (!stopToken.stop_requested()) {
                auto r = _serial.read(std::span(&byte, 1));
                if (r && r.value() == 1 && byte == kCytonHeaderByte) {
                    break;
                }
            }
            continue;
        }

        Sample sample;
        sample.channels.resize(_config.channelCount);
        sample.timestamp = static_cast<double>(buffer[1]);

        for (std::size_t ch = 0; ch < _config.channelCount; ++ch) {
            sample.channels[ch] = parseChannel(&buffer[2 + ch * 3]);
        }

        _ring.push(sample);
    }
}

float OpenBciSource::parseChannel(const std::uint8_t *data)
{
    std::int32_t value =
        (static_cast<std::int32_t>(data[0]) << 16) |
        (static_cast<std::int32_t>(data[1]) << 8) |
        static_cast<std::int32_t>(data[2]);

    if (value & 0x00800000) {
        value |= static_cast<std::int32_t>(0xFF000000);
    }

    return static_cast<float>(value) * kCytonScaleFactor;
}

} // namespace bci::source
