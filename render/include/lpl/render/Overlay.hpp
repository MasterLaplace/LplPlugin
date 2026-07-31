/**
 * @file Overlay.hpp
 * @brief Readout layer: a freestanding text builder, legible glyphs, and the
 *        nearest-neighbour blit that puts a fixed-size frame on any surface.
 *
 * Extracted from the world viewer, where all three had grown as private helpers.
 * None of them is viewer-specific — anything that renders in ring 0 needs a way
 * to say what it is doing, on a surface whose size it does not choose.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_OVERLAY_HPP
#    define LPL_RENDER_OVERLAY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/image/Font8x16.hpp>

namespace lpl::render {

/**
 * @class TextLine
 * @brief Builds one line of readout into a fixed buffer, never past its end.
 *
 * There is no @c snprintf in a freestanding build, and none is needed: a HUD
 * line is text and unsigned numbers. Chaining beats a function with a fixed
 * number of label/value pairs, which is what this replaces — that signature
 * forced callers to pass @c 0u and @c "" as padding whenever they had less to
 * say than it expected.
 */
template <core::u32 Capacity> class TextLine {
public:
    TextLine() noexcept { _buffer[0] = '\0'; }

    /** @brief Resets to empty, so one instance can serve every line of a frame. */
    TextLine &clear() noexcept
    {
        _length = 0u;
        _buffer[0] = '\0';
        return *this;
    }

    TextLine &text(const char *value) noexcept
    {
        for (const char *p = value; *p != '\0' && _length + 1u < Capacity; ++p)
            _buffer[_length++] = *p;
        terminate();
        return *this;
    }

    TextLine &number(core::u32 value) noexcept
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

    /** @brief Signed value with an explicit sign: a world coordinate may be west. */
    TextLine &integer(core::i32 value) noexcept
    {
        if (value < 0)
        {
            text("-");
            return number(static_cast<core::u32>(-value));
        }
        return number(static_cast<core::u32>(value));
    }

    /** @brief One decimal place, which is all a readout ever needs. */
    TextLine &decimal(core::f32 value) noexcept
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

    [[nodiscard]] const char *c_str() const noexcept { return _buffer; }
    [[nodiscard]] core::u32 length() const noexcept { return _length; }

private:
    void terminate() noexcept { _buffer[_length < Capacity ? _length : Capacity - 1u] = '\0'; }

    char _buffer[Capacity]{};
    core::u32 _length{0u};
};

/**
 * @brief Text with a dark shadow one pixel behind it.
 *
 * A near-black clear makes pale text legible by construction, and that is
 * exactly why it hides a bug: add a bright procedural sky and the same text
 * disappears. One offset copy costs a second pass over a few hundred glyph
 * pixels and makes a readout legible over sky, snow and water alike — which is
 * the actual requirement, not "looks fine on the screenshot I happened to take".
 */
inline void drawShadowedText8x16(core::u32 *surface, core::u32 pitchPixels, core::u32 x, core::u32 y, const char *text,
                                 core::u32 colour, core::u32 shadow = 0x00101418u) noexcept
{
    image::drawText8x16(surface, pitchPixels, x + 1u, y + 1u, text, shadow);
    image::drawText8x16(surface, pitchPixels, x, y, text, colour);
}

/**
 * @brief Nearest-neighbour scale of a fixed-size frame onto a display surface.
 *
 * The engine renders at a resolution it picks for its own budget; the display
 * hands back whatever mode the firmware came up in. Deciding the frame size by
 * what the hardware happens to offer would make every performance figure a
 * property of the machine instead of the code.
 */
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

#endif // LPL_RENDER_OVERLAY_HPP
