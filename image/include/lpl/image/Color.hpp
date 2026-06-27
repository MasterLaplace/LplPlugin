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
    core::u16 hue = 0u;        ///< Hue in degrees, 0..359.
    core::u8 saturation = 0u;  ///< Saturation, 0..255.
    core::u8 brightness = 0u;  ///< Brightness (value), 0..255.
};

/** @brief Pack 8-bit channels into 0xAARRGGBB. */
[[nodiscard]] constexpr Rgba packRgba(core::u8 r, core::u8 g, core::u8 b, core::u8 a = 0xFFu) noexcept
{
    return (static_cast<Rgba>(a) << 24) | (static_cast<Rgba>(r) << 16) | (static_cast<Rgba>(g) << 8) |
           static_cast<Rgba>(b);
}

[[nodiscard]] constexpr core::u8 redOf(Rgba c) noexcept { return static_cast<core::u8>((c >> 16) & 0xFFu); }
[[nodiscard]] constexpr core::u8 greenOf(Rgba c) noexcept { return static_cast<core::u8>((c >> 8) & 0xFFu); }
[[nodiscard]] constexpr core::u8 blueOf(Rgba c) noexcept { return static_cast<core::u8>(c & 0xFFu); }
[[nodiscard]] constexpr core::u8 alphaOf(Rgba c) noexcept { return static_cast<core::u8>((c >> 24) & 0xFFu); }

/** @brief Rec.601 integer luminance of an RGB color, 0..255. */
[[nodiscard]] constexpr core::u8 luminanceOf(Rgba c) noexcept
{
    // 0.299 R + 0.587 G + 0.114 B, scaled by 1000 and rounded.
    const core::u32 l = (299u * redOf(c) + 587u * greenOf(c) + 114u * blueOf(c) + 500u) / 1000u;
    return static_cast<core::u8>(l);
}

namespace detail {
[[nodiscard]] constexpr core::u8 max3(core::u8 a, core::u8 b, core::u8 c) noexcept
{
    const core::u8 m = (a > b) ? a : b;
    return (m > c) ? m : c;
}
[[nodiscard]] constexpr core::u8 min3(core::u8 a, core::u8 b, core::u8 c) noexcept
{
    const core::u8 m = (a < b) ? a : b;
    return (m < c) ? m : c;
}
} // namespace detail

/** @brief Convert an RGB color to HSB using integer arithmetic (alpha ignored). */
[[nodiscard]] constexpr Hsb rgbToHsb(Rgba c) noexcept
{
    const core::u8 r = redOf(c);
    const core::u8 g = greenOf(c);
    const core::u8 b = blueOf(c);
    const core::u8 maxv = detail::max3(r, g, b);
    const core::u8 minv = detail::min3(r, g, b);
    const core::u8 delta = static_cast<core::u8>(maxv - minv);

    Hsb out;
    out.brightness = maxv;
    out.saturation = (maxv == 0u) ? 0u : static_cast<core::u8>((static_cast<core::u32>(delta) * 255u) / maxv);

    if (delta == 0u)
    {
        out.hue = 0u; // achromatic
        return out;
    }

    // Hue in degrees, computed *6 then divided, with rounding, kept in [0,360).
    core::i32 hue6;
    if (maxv == r)
        hue6 = (static_cast<core::i32>(g) - static_cast<core::i32>(b)) * 60;
    else if (maxv == g)
        hue6 = (static_cast<core::i32>(b) - static_cast<core::i32>(r)) * 60 + 120 * static_cast<core::i32>(delta);
    else
        hue6 = (static_cast<core::i32>(r) - static_cast<core::i32>(g)) * 60 + 240 * static_cast<core::i32>(delta);

    core::i32 hue = hue6 / static_cast<core::i32>(delta);
    if (hue < 0)
        hue += 360;
    out.hue = static_cast<core::u16>(hue % 360);
    return out;
}

/** @brief Convert an HSB color back to RGB (opaque) using integer arithmetic. */
[[nodiscard]] constexpr Rgba hsbToRgb(const Hsb &hsb) noexcept
{
    const core::u8 v = hsb.brightness;
    if (hsb.saturation == 0u)
        return packRgba(v, v, v);

    const core::u32 h = hsb.hue % 360u;
    const core::u32 region = h / 60u;       // 0..5
    const core::u32 frac = h % 60u;         // 0..59 within the region
    const core::u32 s = hsb.saturation;

    const core::u8 p = static_cast<core::u8>((v * (255u - s)) / 255u);
    const core::u8 q = static_cast<core::u8>((v * (255u - (s * frac) / 60u)) / 255u);
    const core::u8 t = static_cast<core::u8>((v * (255u - (s * (60u - frac)) / 60u)) / 255u);

    switch (region)
    {
        case 0u:  return packRgba(v, t, p);
        case 1u:  return packRgba(q, v, p);
        case 2u:  return packRgba(p, v, t);
        case 3u:  return packRgba(p, q, v);
        case 4u:  return packRgba(t, p, v);
        default:  return packRgba(v, p, q);
    }
}

} // namespace lpl::image

#endif // LPL_IMAGE_COLOR_HPP
