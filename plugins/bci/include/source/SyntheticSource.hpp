/**
 * @file SyntheticSource.hpp
 * @brief Acquisition source producing synthetic EEG via SyntheticGenerator.
 * @author MasterLaplace
 *
 * Wraps SyntheticGenerator behind the ISource interface, providing
 * real-time or burst-mode sample generation without any hardware.
 *
 * @see ISource, SyntheticGenerator
 */

#pragma once

#include "ISource.hpp"
#include "sim/SyntheticGenerator.hpp"

#include <chrono>

namespace bci::source {

/**
 * @brief ISource backed by a deterministic synthetic EEG generator.
 *
 * In real-time mode, the number of samples per read() call is computed
 * from elapsed wall-clock time at the configured sample rate.
 *
 * @code
 *   SyntheticSource src(42, true);
 *   src.start();
 *   std::array<Sample, 256> buf;
 *   auto n = src.read(buf);
 * @endcode
 */
class SyntheticSource final : public ISource {
public:
    /**
     * @brief Constructs a synthetic source.
     *
     * @param seed     PRNG seed (0 = non-deterministic)
     * @param realtime If true, respects the 250 Hz timing; if false, generates
     *                 a fixed batch per read() call (burst mode for tests)
     */
    explicit SyntheticSource(
        std::uint64_t seed = 0,
        bool realtime = true,
        std::size_t channelCount = kDefaultChannelCount,
        float sampleRate = kDefaultSampleRate);

    [[nodiscard]] ExpectedVoid start() override;
    [[nodiscard]] Expected<std::size_t> read(std::span<Sample> buffer) override;
    void stop() noexcept override;
    [[nodiscard]] SourceInfo info() const noexcept override;

    /**
     * @brief Direct access to the underlying generator (e.g. to change profile).
     */
    [[nodiscard]] SyntheticGenerator &generator() noexcept;

private:
    SyntheticGenerator _gen;
    bool _realtime;
    float _sampleRate;
    std::size_t _channelCount;
    bool _running = false;
    std::chrono::steady_clock::time_point _lastRead;
};

} // namespace bci::source
