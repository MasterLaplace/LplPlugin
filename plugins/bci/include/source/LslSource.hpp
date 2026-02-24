/**
 * @file LslSource.hpp
 * @brief Acquisition source via Lab Streaming Layer (LSL) inlet.
 * @author MasterLaplace
 *
 * Connects to a named LSL stream on the local network and reads
 * multi-channel EEG samples in real time. LSL is the de facto standard
 * for BCI data streaming in research settings.
 *
 * @see https://labstreaminglayer.readthedocs.io/
 * @see ISource
 */

#pragma once

#include "ISource.hpp"

#include <lsl_cpp.h>
#include <memory>
#include <string>

namespace bci::source {

/**
 * @brief Configuration for an LSL inlet source.
 */
struct LslSourceConfig {
    std::string streamName = "OpenBCI_EEG";
    double resolveTimeoutSec = 5.0;
};

/**
 * @brief Acquires EEG samples from an LSL stream inlet.
 *
 * The stream is resolved by name on the local network during start().
 * Subsequent read() calls drain available samples without blocking.
 */
class LslSource final : public ISource {
public:
    explicit LslSource(LslSourceConfig config = {});
    ~LslSource() override;

    [[nodiscard]] ExpectedVoid start() override;
    [[nodiscard]] Expected<std::size_t> read(std::span<Sample> buffer) override;
    void stop() noexcept override;
    [[nodiscard]] SourceInfo info() const noexcept override;

private:
    LslSourceConfig _config;
    std::unique_ptr<lsl::stream_inlet> _inlet;
    std::size_t _channelCount = 0;
    double _sampleRate = 0.0;
    bool _running = false;
};

} // namespace bci::source
