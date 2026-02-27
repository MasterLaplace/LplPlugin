/**
 * @file SourceFactory.cpp
 * @brief Implementation of the SourceFactory.
 * @author MasterLaplace
 */

#include "source/SourceFactory.hpp"
#include "source/BrainFlowSource.hpp"
#include "source/CsvReplaySource.hpp"
#include "source/LslSource.hpp"
#include "source/OpenBciSource.hpp"
#include "source/SyntheticSource.hpp"

namespace bci::source {

Expected<std::unique_ptr<ISource>> SourceFactory::create(const SourceConfig &config)
{
    switch (config.mode) {
        case AcquisitionMode::kSerial:
            return std::make_unique<OpenBciSource>(OpenBciConfig{
                .port = config.serialPort,
                .baudRate = kCytonBaudRate,
                .channelCount = config.channelCount
            });

        case AcquisitionMode::kSynthetic:
            return std::make_unique<SyntheticSource>(
                config.syntheticSeed,
                config.syntheticRealtime,
                config.channelCount,
                config.sampleRate);

        case AcquisitionMode::kLsl:
            return std::make_unique<LslSource>(LslSourceConfig{
                .streamName = config.lslStreamName,
                .resolveTimeoutSec = config.lslTimeout
            });

        case AcquisitionMode::kCsvReplay:
            if (config.csvFilePath.empty()) {
                return std::unexpected(
                    Error::make(ErrorCode::kInvalidArgument,
                        "CSV replay mode requires a file path"));
            }
            return std::make_unique<CsvReplaySource>(CsvReplayConfig{
                .filePath = config.csvFilePath,
                .channelCount = config.channelCount,
                .loopback = true,
                .realtime = true,
                .sampleRate = config.sampleRate
            });

        case AcquisitionMode::kBrainFlow:
            return std::make_unique<BrainFlowSource>(BrainFlowConfig{
                .boardId = config.brainFlowBoardId,
                .serialPort = config.brainFlowSerial,
                .serialNumber = config.brainFlowSerialNumber
            });
    }

    return std::unexpected(
        Error::make(ErrorCode::kInvalidArgument, "Unknown acquisition mode"));
}

} // namespace bci::source
