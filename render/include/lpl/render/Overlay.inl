#ifndef LPL_RENDER_OVERLAY_INL
#define LPL_RENDER_OVERLAY_INL

namespace lpl::render {

template <core::u32 Capacity> inline TextLine<Capacity> &TextLine<Capacity>::clear() noexcept
{
    _length = 0u;
    _buffer[0] = '\0';
    return *this;
}

template <core::u32 Capacity> inline TextLine<Capacity> &TextLine<Capacity>::text(const char *value) noexcept
{
    for (const char *p = value; *p != '\0' && _length + 1u < Capacity; ++p)
        _buffer[_length++] = *p;
    terminate();
    return *this;
}

template <core::u32 Capacity> inline TextLine<Capacity> &TextLine<Capacity>::number(core::u32 value) noexcept
{
    char digits[12];
    core::u32 count = 0u;
    do
    {
        digits[count++] = static_cast<char>('0' + (value % 10u));
        value /= 10u;
    } while (value != 0u && count < sizeof(digits));
    while (count != 0u && _length + 1u < Capacity)
        _buffer[_length++] = digits[--count];
    terminate();
    return *this;
}

template <core::u32 Capacity> inline TextLine<Capacity> &TextLine<Capacity>::integer(core::i32 value) noexcept
{
    if (value < 0)
    {
        text("-");
        return number(static_cast<core::u32>(-value));
    }
    return number(static_cast<core::u32>(value));
}

template <core::u32 Capacity> inline TextLine<Capacity> &TextLine<Capacity>::decimal(core::f32 value) noexcept
{
    if (value < 0.0f)
    {
        text("-");
        value = -value;
    }
    const core::u32 whole = static_cast<core::u32>(value);
    const core::u32 tenth = static_cast<core::u32>((value - static_cast<core::f32>(whole)) * 10.0f);
    return number(whole).text(".").number(tenth > 9u ? 9u : tenth);
}

inline void drawShadowedText8x16(core::u32 *surface, core::u32 pitchPixels, core::u32 x, core::u32 y, const char *text,
                                 core::u32 colour, core::u32 shadow) noexcept
{
    image::drawText8x16(surface, pitchPixels, x + 1u, y + 1u, text, shadow);
    image::drawText8x16(surface, pitchPixels, x, y, text, colour);
}

inline void blitScaled(core::u32 *surface, core::u32 pitchPixels, core::u32 surfaceWidth, core::u32 surfaceHeight,
                       const core::u32 *frame, core::u32 frameWidth, core::u32 frameHeight) noexcept
{
    if (surface == nullptr || frame == nullptr || surfaceWidth == 0u || surfaceHeight == 0u)
        return;

    for (core::u32 dy = 0u; dy < surfaceHeight; ++dy)
    {
        const core::u32 *sourceRow = &frame[((dy * frameHeight) / surfaceHeight) * frameWidth];
        core::u32 *destinationRow = &surface[dy * pitchPixels];
        for (core::u32 dx = 0u; dx < surfaceWidth; ++dx)
            destinationRow[dx] = sourceRow[(dx * frameWidth) / surfaceWidth];
    }
}

} // namespace lpl::render

#endif // LPL_RENDER_OVERLAY_INL
