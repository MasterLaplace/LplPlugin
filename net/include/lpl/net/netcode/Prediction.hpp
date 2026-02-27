/**
 * @file Prediction.hpp
 * @brief Client-side prediction buffer for inputs awaiting server ack.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_NETCODE_PREDICTION_HPP
    #define LPL_NET_NETCODE_PREDICTION_HPP

#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <vector>

namespace lpl::net::netcode {

/**
 * @struct PredictedInput
 * @brief An input frame stored for potential re-simulation.
 */
struct PredictedInput
{
    core::u32                   sequence;
    std::vector<core::byte>     data;
};

/**
 * @class Prediction
 * @brief Ring of predicted inputs that have not yet been acknowledged by
 *        the server.
 *
 * Once the server confirms a sequence, all predictions up to and including
 * that sequence are discarded.
 */
class Prediction final : public core::NonCopyable<Prediction>
{
public:
    /**
     * @brief Constructs with a maximum prediction window.
     * @param maxPrediction Maximum unacknowledged frames.
     */
    explicit Prediction(core::u32 maxPrediction = 128);
    ~Prediction();

    /** @brief Stores a new predicted input. */
    void push(PredictedInput input);

    /** @brief Acknowledges all inputs up to @p sequence (inclusive). */
    void acknowledge(core::u32 sequence);

    /** @brief Returns the unacknowledged inputs after @p fromSequence. */
    [[nodiscard]] std::vector<PredictedInput> getUnacknowledged(core::u32 fromSequence) const;

    /** @brief Number of unacknowledged predictions. */
    [[nodiscard]] core::u32 pendingCount() const noexcept;

private:
    core::u32                       _maxPrediction;
    std::vector<PredictedInput>     _buffer;
};

} // namespace lpl::net::netcode

#endif // LPL_NET_NETCODE_PREDICTION_HPP
