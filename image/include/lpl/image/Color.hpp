/**
 * @file Color.hpp
 * @brief Packed RGBA8 color helpers and deterministic RGB<->HSB conversions.
 *
 * Colors are packed into a core::u32 as 0xAARRGGBB. All conversions use integer
 * arithmetic only (no float, no libm) so they are bit-identical across the Linux
 * oracle and the freestanding kernel build — image data is non-authoritative,
 * but keeping it integer-exact avoids any cross-target drift in tests/hashes.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_IMAGE_COLOR_HPP
#    define LPL_IMAGE_COLOR_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::image {

/** @brief A color packed as 0xAARRGGBB. */
using Rgba = core::u32;

/** @brief Hue/Saturation/Brightness triple (hue 0-359, sat/brightness 0-255). */
struct Hsb {
    core::u16 hue = 0u;       /**< Hue in degrees, 0..359. */
    core::u8 saturation = 0u; /**< Saturation, 0..255. */
    core::u8 brightness = 0u; /**< Brightness (value), 0..255. */
};

/**
 * @brief Pack 8-bit channels into 0xAARRGGBB.
 */
[[nodiscard]] constexpr Rgba packRgba(core::u8 r, core::u8 g, core::u8 b, core::u8 a = 0xFFu) noexcept;

[[nodiscard]] constexpr core::u8 redOf(Rgba c) noexcept { return static_cast<core::u8>((c >> 16) & 0xFFu); }
[[nodiscard]] constexpr core::u8 greenOf(Rgba c) noexcept { return static_cast<core::u8>((c >> 8) & 0xFFu); }
[[nodiscard]] constexpr core::u8 blueOf(Rgba c) noexcept { return static_cast<core::u8>(c & 0xFFu); }
[[nodiscard]] constexpr core::u8 alphaOf(Rgba c) noexcept { return static_cast<core::u8>((c >> 24) & 0xFFu); }

/**
 * @brief Rec.601 integer luminance of an RGB color, 0..255.
 *
 * 0.299 R + 0.587 G + 0.114 B, scaled by 1000 and rounded.
 */
[[nodiscard]] constexpr core::u8 luminanceOf(Rgba c) noexcept;

namespace detail {
[[nodiscard]] constexpr core::u8 max3(core::u8 a, core::u8 b, core::u8 c) noexcept;
[[nodiscard]] constexpr core::u8 min3(core::u8 a, core::u8 b, core::u8 c) noexcept;
} // namespace detail

/**
 * @brief Convert an RGB color to HSB using integer arithmetic (alpha ignored).
 *
 * An achromatic color, whose channels are all equal, gets hue 0. Otherwise the hue, in degrees, is
 * computed scaled by the spread between the largest and the smallest channel, divided by it last
 * (truncating, not rounding), and kept in [0,360).
 */
[[nodiscard]] constexpr Hsb rgbToHsb(Rgba c) noexcept;

/**
 * @brief Convert an HSB color back to RGB (opaque) using integer arithmetic.
 *
 * The hue selects one of six 60-degree regions (0..5) and a position within it (0..59).
 */
[[nodiscard]] constexpr Rgba hsbToRgb(const Hsb &hsb) noexcept;

} // namespace lpl::image

#    include "Color.inl"

#endif // LPL_IMAGE_COLOR_HPP
