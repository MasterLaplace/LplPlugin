/**
 * @file CsvReplaySource.cpp
 * @brief Implementation of the CSV file replay acquisition source.
 * @author MasterLaplace
 */

#include "lpl/bci/source/CsvReplaySource.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace lpl::bci::source {

CsvReplaySource::CsvReplaySource(CsvReplayConfig config)
    : _config(std::move(config))
{
}

ExpectedVoid CsvReplaySource::start()
{
    if (_running) {
        return std::unexpected(
            Error::make(ErrorCode::kAlreadyRunning, "CsvReplaySource already running"));
    }

    auto loadResult = loadCsv();
    if (!loadResult) {
        return std::unexpected(loadResult.error());
    }

    _cursor = 0;
    _running = true;
    _lastRead = std::chrono::steady_clock::now();

    return {};
}

Expected<std::size_t> CsvReplaySource::read(std::span<Sample> buffer)
{
    if (!_running) {
        return std::unexpected(
            Error::make(ErrorCode::kNotInitialized, "CsvReplaySource not started"));
    }

    if (_data.empty() || buffer.empty()) {
        return std::size_t{0};
    }

    std::size_t samplesToProcess;
    if (_config.burstMode) {
        samplesToProcess = buffer.size();
    } else if (_config.realtime) {
        auto now = std::chrono::steady_clock::now();
        const float elapsed =
            std::chrono::duration<float>(now - _lastRead).count();
        _lastRead = now;

        samplesToProcess = std::max(
            std::size_t{1},
            static_cast<std::size_t>(elapsed * _config.sampleRate));
        samplesToProcess = std::min(samplesToProcess, kMaxSamplesPerUpdate);
    } else {
        samplesToProcess = kFftUpdateInterval;
    }

    samplesToProcess = std::min(samplesToProcess, buffer.size());
    std::size_t count = 0;

    for (std::size_t t = 0; t < samplesToProcess; ++t) {
        if (_cursor >= _data.size()) {
            if (_config.loopback) {
                _cursor = 0;
            } else {
                _running = false;
            }
            break; // Yield existing samples back to caller, resume next time
        }

        buffer[count].channels = _data[_cursor];
        buffer[count].timestamp = static_cast<double>(_cursor) /
                                  static_cast<double>(_config.sampleRate);
        ++_cursor;
        ++count;
    }

    return count;
}

void CsvReplaySource::stop() noexcept
{
    _running = false;
}

SourceInfo CsvReplaySource::info() const noexcept
{
    return SourceInfo{
        .name = "CSV Replay (" + _config.filePath + ")",
        .channelCount = _channelCount,
        .sampleRate = _config.sampleRate
    };
}

std::size_t CsvReplaySource::totalSamples() const noexcept
{
    return _data.size();
}

std::size_t CsvReplaySource::cursor() const noexcept
{
    return _cursor;
}

Expected<void> CsvReplaySource::loadCsv()
{
    std::ifstream file(_config.filePath);
    if (!file.is_open()) {
        return std::unexpected(
            Error::make(ErrorCode::kFileNotFound, _config.filePath));
    }

    _data.clear();
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') {
            continue;
        }

        std::vector<float> row;
        std::istringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            try {
                row.push_back(std::stof(token));
            } catch (const std::exception &) {
                return std::unexpected(
                    Error::make(ErrorCode::kFileParseError,
                        "Invalid float value '" + token + "' in " + _config.filePath));
            }
        }

        if (!row.empty()) {
            _data.push_back(std::move(row));
        }
    }

    if (_data.empty()) {
        return std::unexpected(
            Error::make(ErrorCode::kEmptyInput,
                "CSV file is empty: " + _config.filePath));
    }

    _channelCount = _data[0].size();

    return {};
}

} // namespace lpl::bci::source
