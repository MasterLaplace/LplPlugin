/**
 * @file NeuralInputState.hpp
 * @brief Processed BCI (neural) input snapshot.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_INPUT_NEURALINPUTSTATE_HPP
#    define LPL_INPUT_NEURALINPUTSTATE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

#    include <array>

namespace lpl::input {

/**
 * @struct NeuralInputState
 * @brief BCI-derived control signals for a single tick.
 *
 * Values are normalised Fixed32 in [0, 1]. The BCI adapter populates this
 * struct each tick from the raw EEG pipeline.
 */
struct NeuralInputState {
    /** @brief Number of control channels. */
    static constexpr core::u32 kChannels = 8;

    std::array<math::Fixed32, kChannels> channels{}; /**< Per-channel activation level (0 = idle, 1 = max intention). */

    math::Fixed32 confidence{}; /**< Confidence score for the overall classification (0 – 1). */

    bool validated{false}; /**< True if the safety chain has validated this frame. */

    core::u32 sequence{0}; /**< Tick sequence number. */
};

} // namespace lpl::input

#endif // LPL_INPUT_NEURALINPUTSTATE_HPP
