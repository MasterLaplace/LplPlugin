/**
 * @file CsvReplaySource.hpp
 * @brief Acquisition source replaying EEG data from a CSV file.
 * @author MasterLaplace
 *
 * Reads a multi-channel CSV (one sample per line, comma-separated float
 * values in µV) and replays it at the original sample rate or in burst
 * mode. Supports loopback for continuous replay.
 *
 * Compatible with exports from OpenBCI GUI, BrainFlow, and any tool
 * producing standard CSV multi-channel EEG recordings.
 *
 * @see ISource
 */

#pragma once

#include "ISource.hpp"
#include "core/Constants.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace bci::source {

/**
 * @brief Configuration for a CSV replay source.
 */
struct CsvReplayConfig {
    std::string filePath;
    std::size_t channelCount = kDefaultChannelCount;
    bool loopback = true;
    bool realtime = true;
    bool burstMode = false;
    float sampleRate = kDefaultSampleRate;
};

/**
 * @brief Replays pre-recorded EEG data from a CSV file.
 *
 * The CSV format expects one sample per line with comma-separated float
 * values. Lines starting with '#' or '%' are treated as comments.
 * Empty lines are skipped.
 *
 * @code
 *   CsvReplaySource src({.filePath = "recording.csv", .loopback = true});
 *   src.start();
 *   std::array<Sample, 256> buf;
 *   auto n = src.read(buf);
 * @endcode
 */
class CsvReplaySource final : public ISource {
public:
    explicit CsvReplaySource(CsvReplayConfig config);

    [[nodiscard]] ExpectedVoid start() override;
    [[nodiscard]] Expected<std::size_t> read(std::span<Sample> buffer) override;
    void stop() noexcept override;
    [[nodiscard]] SourceInfo info() const noexcept override;

    /**
     * @brief Returns the total number of samples in the loaded file.
     */
    [[nodiscard]] std::size_t totalSamples() const noexcept;

    /**
     * @brief Returns the current read cursor position.
     */
    [[nodiscard]] std::size_t cursor() const noexcept;

private:
    Expected<void> loadCsv();

    CsvReplayConfig _config;
    std::vector<std::vector<float>> _data;
    std::size_t _cursor = 0;
    std::size_t _channelCount = 0;
    bool _running = false;
    std::chrono::steady_clock::time_point _lastRead;
};

} // namespace bci::source
