/**
 * @file SourceFactory.hpp
 * @brief Abstract Factory for creating ISource instances from configuration.
 * @author MasterLaplace
 *
 * Encapsulates the construction logic for all acquisition backends.
 * Unlike V1, there is no silent fallback to Synthetic — if the requested
 * source fails, an explicit error is returned to the caller.
 *
 * @see ISource, AcquisitionMode
 */

#pragma once

#include "ISource.hpp"

#include <memory>
#include <string>

namespace lpl::bci::source {

/**
 * @brief Unified configuration for any acquisition source.
 */
struct SourceConfig {
    AcquisitionMode mode = AcquisitionMode::kSynthetic;
    std::size_t channelCount = 8;
    float sampleRate = 250.0f;
    std::string serialPort = "/dev/ttyUSB0";
    std::string csvFilePath;
    std::string lslStreamName = "OpenBCI_EEG";
    double lslTimeout = 5.0;
    std::uint64_t syntheticSeed = 0;
    bool syntheticRealtime = true;
    int brainFlowBoardId = -1;
    std::string brainFlowSerial;
    std::string brainFlowSerialNumber;
};

/**
 * @brief Factory that instantiates the appropriate ISource from a SourceConfig.
 *
 * @code
 *   SourceConfig cfg{.mode = AcquisitionMode::kSynthetic, .syntheticSeed = 42};
 *   auto source = SourceFactory::create(cfg);
 *   if (source) {
 *       auto result = source.value()->start();
 *   }
 * @endcode
 */
class SourceFactory {
public:
    SourceFactory() = delete;

    /**
     * @brief Creates an ISource for the requested acquisition mode.
     *
     * @param config Source configuration (mode, port, file, etc.)
     * @return A unique_ptr to the source on success, or an Error
     */
    [[nodiscard]] static Expected<std::unique_ptr<ISource>> create(const SourceConfig &config);
};

} // namespace lpl::bci::source
