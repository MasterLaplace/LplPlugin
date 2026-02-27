/**
 * @file ISource.hpp
 * @brief Abstract interface for EEG data acquisition sources.
 * @author MasterLaplace
 *
 * Every acquisition backend (hardware serial, LSL, BrainFlow, CSV replay,
 * synthetic generator) implements this interface. Sources are responsible
 * only for raw sample acquisition — signal processing is delegated to the
 * DSP Pipeline (Single Responsibility Principle).
 *
 * @see SourceFactory, Pipeline
 */

#pragma once

#include "lpl/bci/core/Error.hpp"
#include "lpl/bci/core/Types.hpp"

#include <span>

namespace lpl::bci::source {

/**
 * @brief Abstract acquisition source for multi-channel EEG data.
 *
 * Contract:
 * 1. start() initializes hardware/connection. Must be called before read().
 * 2. read() fills the provided buffer with available samples (non-blocking).
 * 3. stop() releases all resources (RAII — also called by the destructor).
 * 4. info() returns metadata about the source.
 *
 * Sources do NOT perform signal processing (no FFT, no metrics).
 * They produce raw Sample values that flow into the DSP Pipeline.
 */
class ISource {
public:
    virtual ~ISource() = default;

    ISource(const ISource &) = delete;
    ISource &operator=(const ISource &) = delete;
    ISource(ISource &&) = default;
    ISource &operator=(ISource &&) = default;

    /**
     * @brief Initializes the acquisition backend.
     *
     * @return void on success, or an Error describing the failure
     */
    [[nodiscard]] virtual ExpectedVoid start() = 0;

    /**
     * @brief Reads available samples into the provided buffer.
     *
     * @param buffer Span of Sample objects to fill
     * @return Number of samples actually read, or an Error
     */
    [[nodiscard]] virtual Expected<std::size_t> read(std::span<Sample> buffer) = 0;

    /**
     * @brief Stops acquisition and releases all resources.
     */
    virtual void stop() noexcept = 0;

    /**
     * @brief Returns metadata about this source.
     */
    [[nodiscard]] virtual SourceInfo info() const noexcept = 0;

protected:
    ISource() = default;
};

} // namespace lpl::bci::source
