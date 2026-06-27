/**
 * @file Image.cpp
 * @brief Out-of-line Image sampling + histogram (integer-only, deterministic).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */
#include <lpl/image/Image.hpp>

namespace lpl::image {

namespace {

/// Blend two channel values with an 0..256 weight toward @p hi.
[[nodiscard]] core::u8 lerp8(core::u8 lo, core::u8 hi, core::u32 weight) noexcept
{
    return static_cast<core::u8>((static_cast<core::u32>(lo) * (256u - weight) + static_cast<core::u32>(hi) * weight) >>
                                 8);
}

/// Bilinearly blend four corner colors with 0..256 weights (wx toward right,
/// wy toward bottom), per channel.
[[nodiscard]] Rgba blend4(Rgba c00, Rgba c10, Rgba c01, Rgba c11, core::u32 wx, core::u32 wy) noexcept
{
    const core::u8 top_r = lerp8(redOf(c00), redOf(c10), wx);
    const core::u8 top_g = lerp8(greenOf(c00), greenOf(c10), wx);
    const core::u8 top_b = lerp8(blueOf(c00), blueOf(c10), wx);
    const core::u8 top_a = lerp8(alphaOf(c00), alphaOf(c10), wx);

    const core::u8 bot_r = lerp8(redOf(c01), redOf(c11), wx);
    const core::u8 bot_g = lerp8(greenOf(c01), greenOf(c11), wx);
    const core::u8 bot_b = lerp8(blueOf(c01), blueOf(c11), wx);
    const core::u8 bot_a = lerp8(alphaOf(c01), alphaOf(c11), wx);

    return packRgba(lerp8(top_r, bot_r, wy), lerp8(top_g, bot_g, wy), lerp8(top_b, bot_b, wy),
                    lerp8(top_a, bot_a, wy));
}

} // namespace

Rgba Image::sampleBilinear(core::u32 u, core::u32 v) const noexcept
{
    if (_width == 0u || _height == 0u)
        return 0u;

    // Map normalized Q16 to a Q16 pixel coordinate biased by -0.5 px so samples
    // land at texel centres, then split into integer index + 0..256 weight.
    const core::i64 fx = (static_cast<core::i64>(u) * _width) - 0x8000;
    const core::i64 fy = (static_cast<core::i64>(v) * _height) - 0x8000;

    const core::i32 x0 = static_cast<core::i32>(fx >> 16);
    const core::i32 y0 = static_cast<core::i32>(fy >> 16);
    const core::u32 wx = static_cast<core::u32>((fx >> 8) & 0xFFu);
    const core::u32 wy = static_cast<core::u32>((fy >> 8) & 0xFFu);

    return blend4(at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1), wx, wy);
}

Histogram Image::histogram() const noexcept
{
    Histogram out;
    for (const Rgba pixel : _pixels)
    {
        ++out.red[redOf(pixel)];
        ++out.green[greenOf(pixel)];
        ++out.blue[blueOf(pixel)];
        ++out.luminance[luminanceOf(pixel)];
    }
    return out;
}

} // namespace lpl::image
