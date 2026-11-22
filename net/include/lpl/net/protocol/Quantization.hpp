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
 * @brief Whether [@p minimum, @p maximum] is a range a value can be quantized over.
 *
 * @details The maximum exceeds the minimum, and their distance is a finite float: an empty, reversed
 *          or NaN range, or one wider than the largest float, whose share would be inf / inf, is not.
 *          The writer and the reader both check it, since a range can come from the wire.
 *
 * @param minimum Lower bound of the range.
 * @param maximum Upper bound of the range.
 * @return Whether it is a range.
 */
[[nodiscard]] constexpr bool isQuantizationRange(float minimum, float maximum) noexcept;

/**
 * @brief The step of [0, 2^bits - 1] that @p value takes over the range [@p minimum, @p maximum].
 *
 * @details The value is clamped to the range, then mapped to the nearest step. NaN takes step 0, as a
 *          value below the range does, on every target: no NaN reaches the conversion to an integer, whose
 *          result the language leaves undefined and which x86-64 and i686 answer differently.
 *          The top of the range takes the top step on every width, 32 bits included, where the
 *          float of the top step is 2^32 and would not convert.
 *
 * @param value   Value to quantize.
 * @param minimum Lower bound of the range.
 * @param maximum Upper bound of the range, which the caller has checked with isQuantizationRange().
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
 * @brief @p radians wrapped into one turn, so any real angle quantizes.
 *
 * @details The result is in [0, 2*pi], 2*pi included: an angle a hair below a whole number of turns
 *          rounds to it (-2.4e-7 rad comes back as 2*pi). Past a great many turns, the rounding of the
 *          turns themselves can leave it outside [0, 2*pi] (1e9 rad comes back as -57.7), and quantize()
 *          clamps it to the range. An angle whose count of turns does not fit 32 bits, infinity or NaN
 *          comes back as it is, and quantize() clamps it too: counting its turns would convert out of the
 *          range of an integer.
 *
 * @param radians An angle.
 * @return The angle within one turn as rounding leaves it, or @p radians when its turns cannot be counted.
 */
[[nodiscard]] inline float wrapAngle(float radians) noexcept;

} // namespace lpl::net::protocol

#    include "Quantization.inl"

#endif /* LPL_NET_PROTOCOL_QUANTIZATION_HPP */
