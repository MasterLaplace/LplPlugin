/**
 * @file Quantization.hpp
 * @brief The steps a float takes on the wire when it is quantized (book §6.3.3).
 *
 * Non-authoritative wire data only: the quantization is lossy, so a quantized value never flows back
 * into authoritative Fixed32. The float arithmetic is the one the determinism contract allows on such
 * paths: IEEE single precision through SSE, without contraction, the same on the host and in ring 0.
 *
 * @author MasterLaplace
 * @version 0.0.0
 * @date 2026-11-22
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_NET_PROTOCOL_QUANTIZATION_HPP
#    define LPL_NET_PROTOCOL_QUANTIZATION_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::net::protocol {

/** A full turn, in radians: the range an angle is wrapped into. */
inline constexpr float kTwoPi = 6.28318530717958647692f;

/**
 * @brief The largest integer @p bits bits hold: the top step of a quantization on that width.
 *
 * @param bits A width in 1..32.
 * @return 2^bits - 1.
 */
[[nodiscard]] constexpr core::u32 quantizationCeiling(core::u32 bits) noexcept;

/**
 * @brief The step of [0, 2^bits - 1] that @p value takes over the range [@p minimum, @p maximum].
 *
 * @details The value is clamped to the range, then mapped to the nearest step.
 *
 * @param value   Value to quantize.
 * @param minimum Lower bound of the range.
 * @param maximum Upper bound of the range, which the caller has checked exceeds @p minimum.
 * @param bits    Width of the step, which the caller has checked is in 1..32.
 * @return The step.
 */
[[nodiscard]] inline core::u32 quantize(float value, float minimum, float maximum, core::u32 bits) noexcept;

/**
 * @brief The value step @p step stands for over the range [@p minimum, @p maximum].
 *
 * @param step    A step of [0, 2^bits - 1].
 * @param minimum Lower bound of the range.
 * @param maximum Upper bound of the range.
 * @param bits    Width of the step, which the caller has checked is in 1..32.
 * @return The value: @p minimum for step 0, @p maximum for the top step.
 */
[[nodiscard]] inline float dequantize(core::u32 step, float minimum, float maximum, core::u32 bits) noexcept;

/**
 * @brief @p radians wrapped into the turn [0, 2*pi), so any real angle quantizes.
 *
 * @param radians An angle.
 * @return The same angle within one turn.
 */
[[nodiscard]] inline float wrapAngle(float radians) noexcept;

} // namespace lpl::net::protocol

#    include "Quantization.inl"

#endif /* LPL_NET_PROTOCOL_QUANTIZATION_HPP */
