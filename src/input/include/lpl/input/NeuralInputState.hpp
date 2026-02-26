// /////////////////////////////////////////////////////////////////////////////
/// @file NeuralInputState.hpp
/// @brief Processed BCI (neural) input snapshot.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/math/FixedPoint.hpp>
#include <lpl/core/Types.hpp>

#include <array>

namespace lpl::input {

// /////////////////////////////////////////////////////////////////////////////
/// @struct NeuralInputState
/// @brief BCI-derived control signals for a single tick.
///
/// Values are normalised Fixed32 in [0, 1]. The BCI adapter populates this
/// struct each tick from the raw EEG pipeline.
// /////////////////////////////////////////////////////////////////////////////
struct NeuralInputState
{
    /// @brief Number of control channels.
    static constexpr core::u32 kChannels = 8;

    /// @brief Per-channel activation level (0 = idle, 1 = max intention).
    std::array<math::Fixed32, kChannels> channels{};

    /// @brief Confidence score for the overall classification (0 – 1).
    math::Fixed32 confidence{};

    /// @brief True if the safety chain has validated this frame.
    bool validated{false};

    /// @brief Tick sequence number.
    core::u32 sequence{0};
};

} // namespace lpl::input
