/**
 * @file BrainFlowSource.hpp
 * @brief Acquisition source via the BrainFlow SDK (20+ board support).
 * @author MasterLaplace
 *
 * BrainFlow provides a hardware-agnostic API supporting OpenBCI (Cyton,
 * Ganglion), Muse, Neurosity Crown, and many other EEG headsets.
 *
 * @see https://brainflow.readthedocs.io/
 * @see ISource
 */

#pragma once

#include "ISource.hpp"

#ifdef LPL_HAS_BRAINFLOW
#include <board_shim.h>
#endif

#include <memory>
#include <string>
#include <vector>

namespace bci::source {

/**
 * @brief Configuration for a BrainFlow acquisition source.
 */
struct BrainFlowConfig {
    int boardId = -1;
    std::string serialPort;
    std::string serialNumber;
};

/**
 * @brief Acquires EEG via BrainFlow's board-agnostic API.
 *
 * Supports all boards registered in the BrainFlow SDK.
 * Requires LPL_HAS_BRAINFLOW to be defined at compile time.
 */
class BrainFlowSource final : public ISource {
public:
    explicit BrainFlowSource(BrainFlowConfig config = {});
    ~BrainFlowSource() override;

    [[nodiscard]] ExpectedVoid start() override;
    [[nodiscard]] Expected<std::size_t> read(std::span<Sample> buffer) override;
    void stop() noexcept override;
    [[nodiscard]] SourceInfo info() const noexcept override;

private:
    BrainFlowConfig _config;
    bool _running = false;

#ifdef LPL_HAS_BRAINFLOW
    std::unique_ptr<BoardShim> _board;
    std::vector<int> _eegChannels;
    int _sampleRate = 250;
#endif
};

} // namespace bci::source
