/**
 * @file Morton.hpp
 * @brief Z-order curve encoding / decoding via bit-interleaving.
 *
 * Converts 2D/3D integer coordinates to a single Morton code that
 * preserves spatial locality.  A bias of 2^20 is applied to support
 * negative coordinates without branching.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MATH_MORTON_HPP
#    define LPL_MATH_MORTON_HPP

#    include <lpl/core/Constants.hpp>
#    include <lpl/core/Platform.hpp>
#    include <lpl/core/Types.hpp>

namespace lpl::math::morton {

/**
 * @brief Encode a 2D coordinate pair into a 32-bit Morton code.
 * @param x First axis (biased by kMortonBias internally).
 * @param y Second axis (biased by kMortonBias internally).
 * @return 32-bit Morton key.
 */
[[nodiscard]] LPL_HD constexpr core::u32 encode2D(core::i32 x, core::i32 y);

/**
 * @brief Encode a 3D coordinate triple into a 63-bit Morton code.
 * @param x First axis.
 * @param y Second axis.
 * @param z Third axis.
 * @return 63-bit Morton key stored in a u64.
 */
[[nodiscard]] LPL_HD constexpr core::u64 encode3D(core::i32 x, core::i32 y, core::i32 z);

/**
 * @brief Encode three ALREADY non-negative grid coordinates, applying no bias.
 *
 * @ref encode3D exists for world coordinates that straddle the origin: it adds
 * @c kMortonBias so a negative axis still lands in the unsigned grid. That makes
 * it the wrong tool for a caller that has already mapped its coordinates into a
 * bounded grid, because the bias is added a SECOND time and the sum wraps the
 * 21-bit field — which is exactly what an octree that reads the key's high bits
 * as an octant path cannot survive. Such a caller wants this overload.
 *
 * @param x,y,z Grid coordinates, each clamped by the caller to [0, 2^21).
 * @return 63-bit Morton key.
 */
[[nodiscard]] LPL_HD constexpr core::u64 encode3DGrid(core::u32 x, core::u32 y, core::u32 z);

/**
 * @brief Decode a 2D Morton code back into its coordinate pair.
 * @param code 32-bit Morton key.
 * @param[out] x Decoded first axis.
 * @param[out] y Decoded second axis.
 */
LPL_HD constexpr void decode2D(core::u32 code, core::i32 &x, core::i32 &y);

/**
 * @brief Decode a 3D Morton code back into its coordinate triple.
 * @param code 63-bit Morton key.
 * @param[out] x Decoded first axis.
 * @param[out] y Decoded second axis.
 * @param[out] z Decoded third axis.
 */
LPL_HD constexpr void decode3D(core::u64 code, core::i32 &x, core::i32 &y, core::i32 &z);

namespace detail {

/**
 * @brief Spread the low 16 bits of @p n to the even bits of a 32-bit value.
 */
[[nodiscard]] LPL_HD constexpr core::u32 part1by1(core::u32 n);

/**
 * @brief Spread the low 21 bits of @p n to every third bit of a 63-bit value.
 */
[[nodiscard]] LPL_HD constexpr core::u64 part1by2(core::u64 n);

/**
 * @brief Inverse of part1by1: compact every other bit into a 16-bit value.
 */
[[nodiscard]] LPL_HD constexpr core::u32 compact1by1(core::u32 v);

/**
 * @brief Inverse of part1by2: compact every third bit into a 21-bit value.
 */
[[nodiscard]] LPL_HD constexpr core::u64 compact1by2(core::u64 v);

} // namespace detail

} // namespace lpl::math::morton

#    include "Morton.inl"

#endif // LPL_MATH_MORTON_HPP
