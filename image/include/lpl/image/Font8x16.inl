#ifndef LPL_IMAGE_FONT8X16_INL
#define LPL_IMAGE_FONT8X16_INL

namespace lpl::image {

inline void drawGlyph8x16(core::u32 *destination, core::u32 pitchPixels, core::u32 x, core::u32 y,
                          unsigned char character, core::u32 colorRgb) noexcept
{
    const core::u8 *glyph = kFont8x16[character];
    for (core::u32 row = 0u; row < kFontHeight; ++row)
    {
        const core::u8 bits = glyph[row];
        core::u32 *destinationRow = &destination[(y + row) * pitchPixels + x];
        for (core::u32 column = 0u; column < kFontWidth; ++column)
        {
            if (bits & (0x80u >> column))
                destinationRow[column] = colorRgb;
        }
    }
}

inline void drawText8x16(core::u32 *destination, core::u32 pitchPixels, core::u32 x, core::u32 y, const char *text,
                         core::u32 colorRgb) noexcept
{
    for (core::u32 i = 0u; text[i] != '\0'; ++i)
        drawGlyph8x16(destination, pitchPixels, x + i * kFontWidth, y, static_cast<unsigned char>(text[i]), colorRgb);
}

} // namespace lpl::image

#endif // LPL_IMAGE_FONT8X16_INL
