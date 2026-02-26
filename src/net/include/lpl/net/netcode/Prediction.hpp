// /////////////////////////////////////////////////////////////////////////////
/// @file Prediction.hpp
/// @brief Client-side prediction buffer for inputs awaiting server ack.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <vector>

namespace lpl::net::netcode {

// /////////////////////////////////////////////////////////////////////////////
/// @struct PredictedInput
/// @brief An input frame stored for potential re-simulation.
// /////////////////////////////////////////////////////////////////////////////
struct PredictedInput
{
    core::u32                   sequence;
    std::vector<core::byte>     data;
};

// /////////////////////////////////////////////////////////////////////////////
/// @class Prediction
/// @brief Ring of predicted inputs that have not yet been acknowledged by
///        the server.
///
/// Once the server confirms a sequence, all predictions up to and including
/// that sequence are discarded.
// /////////////////////////////////////////////////////////////////////////////
class Prediction final : public core::NonCopyable<Prediction>
{
public:
    /// @brief Constructs with a maximum prediction window.
    /// @param maxPrediction Maximum unacknowledged frames.
    explicit Prediction(core::u32 maxPrediction = 128);
    ~Prediction();

    /// @brief Stores a new predicted input.
    void push(PredictedInput input);

    /// @brief Acknowledges all inputs up to @p sequence (inclusive).
    void acknowledge(core::u32 sequence);

    /// @brief Returns the unacknowledged inputs after @p fromSequence.
    [[nodiscard]] std::vector<PredictedInput> getUnacknowledged(core::u32 fromSequence) const;

    /// @brief Number of unacknowledged predictions.
    [[nodiscard]] core::u32 pendingCount() const noexcept;

private:
    core::u32                       maxPrediction_;
    std::vector<PredictedInput>     buffer_;
};

} // namespace lpl::net::netcode
