#ifndef LPL_IMAGE_COLOR_INL
#define LPL_IMAGE_COLOR_INL

namespace lpl::image {

constexpr Rgba packRgba(core::u8 r, core::u8 g, core::u8 b, core::u8 a) noexcept
{
    return (static_cast<Rgba>(a) << 24) | (static_cast<Rgba>(r) << 16) | (static_cast<Rgba>(g) << 8) |
           static_cast<Rgba>(b);
}

constexpr core::u8 luminanceOf(Rgba c) noexcept
{
    const core::u32 l = (299u * redOf(c) + 587u * greenOf(c) + 114u * blueOf(c) + 500u) / 1000u;
    return static_cast<core::u8>(l);
}

namespace detail {
constexpr core::u8 max3(core::u8 a, core::u8 b, core::u8 c) noexcept
{
    const core::u8 m = (a > b) ? a : b;
    return (m > c) ? m : c;
}
constexpr core::u8 min3(core::u8 a, core::u8 b, core::u8 c) noexcept
{
    const core::u8 m = (a < b) ? a : b;
    return (m < c) ? m : c;
}
} // namespace detail

constexpr Hsb rgbToHsb(Rgba c) noexcept
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
        out.hue = 0u;
        return out;
    }

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

constexpr Rgba hsbToRgb(const Hsb &hsb) noexcept
{
    const core::u8 v = hsb.brightness;
    if (hsb.saturation == 0u)
        return packRgba(v, v, v);

    const core::u32 h = hsb.hue % 360u;
    const core::u32 region = h / 60u;
    const core::u32 frac = h % 60u;
    const core::u32 s = hsb.saturation;

    const core::u8 p = static_cast<core::u8>((v * (255u - s)) / 255u);
    const core::u8 q = static_cast<core::u8>((v * (255u - (s * frac) / 60u)) / 255u);
    const core::u8 t = static_cast<core::u8>((v * (255u - (s * (60u - frac)) / 60u)) / 255u);

    switch (region)
    {
    case 0u: return packRgba(v, t, p);
    case 1u: return packRgba(q, v, p);
    case 2u: return packRgba(p, v, t);
    case 3u: return packRgba(p, q, v);
    case 4u: return packRgba(t, p, v);
    default: return packRgba(v, p, q);
    }
}

} // namespace lpl::image

#endif // LPL_IMAGE_COLOR_INL
