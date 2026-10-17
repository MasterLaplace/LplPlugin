#ifndef LPL_RENDER_MIP_TEXTURE_INL
#define LPL_RENDER_MIP_TEXTURE_INL

namespace lpl::render {

inline MipTexture::MipTexture(const Texture &base, core::u32 maxLevels)
{
    _levels.push_back(base);
    while (_levels.size() < maxLevels &&
           (_levels[_levels.size() - 1u].width() > 1u || _levels[_levels.size() - 1u].height() > 1u))
        _levels.push_back(_levels[_levels.size() - 1u].halved());
}

inline core::u32 MipTexture::sample(core::u32 uQ16, core::u32 vQ16, core::f32 level) const noexcept
{
    if (_levels.empty())
        return 0u;
    if (level < 0.0f)
        level = 0.0f;
    const core::u32 last = static_cast<core::u32>(_levels.size()) - 1u;
    const core::u32 low = static_cast<core::u32>(level);
    if (low >= last)
        return _levels[last].sampleBilinear(uQ16, vQ16);

    const core::u32 a = _levels[low].sampleBilinear(uQ16, vQ16);
    const core::u32 b = _levels[low + 1u].sampleBilinear(uQ16, vQ16);
    const core::u32 t = static_cast<core::u32>((level - static_cast<core::f32>(low)) * 256.0f);

    core::u32 out = 0u;
    for (core::u32 shift = 0u; shift <= 16u; shift += 8u)
    {
        const core::u32 from = (a >> shift) & 0xFFu;
        const core::u32 to = (b >> shift) & 0xFFu;
        out |= (((from * (256u - t) + to * t) >> 8) & 0xFFu) << shift;
    }
    return out;
}

inline core::f32 MipTexture::levelForFootprint(core::f32 texelsPerPixel) noexcept
{
    if (texelsPerPixel <= 1.0f)
        return 0.0f;
    core::u32 bits = 0u;
    __builtin_memcpy(&bits, &texelsPerPixel, sizeof(bits));
    const core::i32 exponent = static_cast<core::i32>((bits >> 23) & 0xFFu) - 127;
    const core::f32 mantissa = static_cast<core::f32>(bits & 0x7FFFFFu) * (1.0f / 8388608.0f);
    return static_cast<core::f32>(exponent) + mantissa;
}

} // namespace lpl::render

#endif // LPL_RENDER_MIP_TEXTURE_INL
