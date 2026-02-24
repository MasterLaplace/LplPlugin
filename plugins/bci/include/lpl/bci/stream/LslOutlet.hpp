/**
 * @file LslOutlet.hpp
 * @brief Lab Streaming Layer outlet for broadcasting BCI data.
 * @author MasterLaplace
 *
 * Wraps liblsl outlet functionality to broadcast NeuralState or raw samples
 * over the network. Other applications (OpenViBE, custom visualizers) can
 * subscribe to this stream.
 *
 * @see LslSource
 */

#pragma once

#include "lpl/bci/core/Error.hpp"
#include "lpl/bci/core/Types.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace lpl::bci::stream {

/**
 * @brief Configuration for an LSL outlet stream.
 */
struct LslOutletConfig {
    std::string streamName = "LplBciOutlet";
    std::string streamType = "EEG";
    std::size_t channelCount = 8;
    float sampleRate = 250.0f;
};

/**
 * @brief RAII wrapper around a liblsl outlet.
 *
 * Movable, non-copyable. The underlying lsl::stream_outlet is created
 * on `open()` and destroyed when this object is destructed.
 */
class LslOutlet {
public:
    LslOutlet() = default;

    /**
     * @brief Opens the outlet with the given configuration.
     *
     * @param config Stream parameters
     * @return Error on failure
     */
    [[nodiscard]] ExpectedVoid open(const LslOutletConfig& config);

    /**
     * @brief Pushes a single multi-channel sample.
     *
     * @param data Sample values (size >= channelCount)
     */
    void pushSample(std::span<const float> data) noexcept;

    /**
     * @brief Pushes a NeuralState as a sample (alpha + beta + means).
     *
     * Serializes all channel alphas, then all channel betas, then mean
     * alpha and mean beta as trailing values.
     *
     * @param state Neural state to broadcast
     */
    void pushNeuralState(const NeuralState& state) noexcept;

    /**
     * @brief Checks if the outlet is currently open.
     */
    [[nodiscard]] bool isOpen() const noexcept;

    /**
     * @brief Closes the outlet.
     */
    void close() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::bci::stream
