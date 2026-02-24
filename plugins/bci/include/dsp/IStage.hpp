/**
 * @file IStage.hpp
 * @brief Abstract interface for a single stage in the DSP processing pipeline.
 * @author MasterLaplace
 *
 * Every signal processing operation (windowing, FFT, band extraction, etc.)
 * implements this interface. Stages are composed sequentially by the Pipeline
 * class using the Builder pattern.
 *
 * @see Pipeline
 */

#pragma once

#include "lpl/bci/core/Error.hpp"
#include "lpl/bci/core/Types.hpp"

#include <memory>
#include <string_view>

namespace bci::dsp {

/**
 * @brief A single processing stage that transforms a SignalBlock.
 *
 * Contract:
 *  - process() must not modify the input block.
 *  - process() returns a new SignalBlock on success, or an Error.
 *  - name() returns a stable, non-empty identifier for diagnostics.
 */
class IStage {
public:
    virtual ~IStage() = default;

    IStage(const IStage &) = delete;
    IStage &operator=(const IStage &) = delete;
    IStage(IStage &&) = default;
    IStage &operator=(IStage &&) = default;

    /**
     * @brief Transforms the input signal block.
     *
     * @param input The signal block to process (not modified)
     * @return A new SignalBlock on success, or an Error describing the failure
     */
    [[nodiscard]] virtual Expected<SignalBlock> process(const SignalBlock &input) = 0;

    /**
     * @brief Returns a human-readable name for this stage.
     */
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

protected:
    IStage() = default;
};

} // namespace bci::dsp
