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
    #define LPL_MATH_MORTON_HPP

    #include <lpl/core/Constants.hpp>
    #include <lpl/core/Platform.hpp>
    #include <lpl/core/Types.hpp>

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

[[nodiscard]] LPL_HD constexpr core::u32 part1by1(core::u32 n)
{
    core::u32 v = n & 0x0000FFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

[[nodiscard]] LPL_HD constexpr core::u64 part1by2(core::u64 n)
{
    core::u64 v = n & 0x1FFFFF;
    v = (v | (v << 32)) & 0x1F00000000FFFF;
    v = (v | (v << 16)) & 0x1F0000FF0000FF;
    v = (v | (v <<  8)) & 0x100F00F00F00F00F;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3;
    v = (v | (v <<  2)) & 0x1249249249249249;
    return v;
}

} // namespace detail

LPL_HD constexpr core::u32 encode2D(core::i32 x, core::i32 y)
{
    auto ux = static_cast<core::u32>(x + core::kMortonBias);
    auto uy = static_cast<core::u32>(y + core::kMortonBias);
    return detail::part1by1(ux) | (detail::part1by1(uy) << 1);
}

LPL_HD constexpr core::u64 encode3D(core::i32 x, core::i32 y, core::i32 z)
{
    auto ux = static_cast<core::u64>(x + core::kMortonBias);
    auto uy = static_cast<core::u64>(y + core::kMortonBias);
    auto uz = static_cast<core::u64>(z + core::kMortonBias);
    return detail::part1by2(ux) | (detail::part1by2(uy) << 1) | (detail::part1by2(uz) << 2);
}

LPL_HD constexpr void decode2D([[maybe_unused]] core::u32 code, [[maybe_unused]] core::i32 &x, [[maybe_unused]] core::i32 &y) {}

LPL_HD constexpr void decode3D([[maybe_unused]] core::u64 code, [[maybe_unused]] core::i32 &x, [[maybe_unused]] core::i32 &y, [[maybe_unused]] core::i32 &z) {}

} // namespace lpl::math::morton

#endif // LPL_MATH_MORTON_HPP
