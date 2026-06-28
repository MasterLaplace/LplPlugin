/**
 * @file Codec.cpp
 * @brief Portable PPM (P6) encode/decode — deterministic, freestanding-safe.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */
#include <lpl/image/Codec.hpp>

namespace lpl::image {

namespace {

/// Append the decimal digits of @p value to @p out.
void appendUint(pmr::vector<core::u8> &out, core::u32 value)
{
    char digits[10];
    core::i32 count = 0;
    do
    {
        digits[count++] = static_cast<char>('0' + (value % 10u));
        value /= 10u;
    } while (value != 0u);
    while (count-- > 0)
        out.push_back(static_cast<core::u8>(digits[count]));
}

[[nodiscard]] bool isSpace(core::u8 c) noexcept
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

/// Parser cursor over a PPM header.
struct Cursor {
    const core::u8 *data;
    core::usize length;
    core::usize pos;
};

/// Skip whitespace and '#' comment lines.
void skipBlanks(Cursor &c) noexcept
{
    while (c.pos < c.length)
    {
        const core::u8 ch = c.data[c.pos];
        if (isSpace(ch))
            ++c.pos;
        else if (ch == '#')
            while (c.pos < c.length && c.data[c.pos] != '\n')
                ++c.pos;
        else
            break;
    }
}

/// Parse a non-negative decimal integer; false if no digits were found.
[[nodiscard]] bool parseUint(Cursor &c, core::u32 &out) noexcept
{
    skipBlanks(c);
    if (c.pos >= c.length || c.data[c.pos] < '0' || c.data[c.pos] > '9')
        return false;
    core::u32 value = 0u;
    while (c.pos < c.length && c.data[c.pos] >= '0' && c.data[c.pos] <= '9')
    {
        value = value * 10u + static_cast<core::u32>(c.data[c.pos] - '0');
        ++c.pos;
    }
    out = value;
    return true;
}

} // namespace

bool writePpm(const Image &image, pmr::vector<core::u8> &out)
{
    if (image.empty())
        return false;

    out.clear();
    out.push_back('P');
    out.push_back('6');
    out.push_back('\n');
    appendUint(out, image.width());
    out.push_back(' ');
    appendUint(out, image.height());
    out.push_back('\n');
    appendUint(out, 255u);
    out.push_back('\n');

    for (core::u32 y = 0u; y < image.height(); ++y)
        for (core::u32 x = 0u; x < image.width(); ++x)
        {
            const Rgba c = image.at(static_cast<core::i32>(x), static_cast<core::i32>(y));
            out.push_back(redOf(c));
            out.push_back(greenOf(c));
            out.push_back(blueOf(c));
        }
    return true;
}

bool readPpm(const core::u8 *data, core::usize length, Image &out)
{
    if (data == nullptr || length < 2u || data[0] != 'P' || data[1] != '6')
        return false;

    Cursor c{data, length, 2u};
    core::u32 width = 0u;
    core::u32 height = 0u;
    core::u32 maxval = 0u;
    if (!parseUint(c, width) || !parseUint(c, height) || !parseUint(c, maxval))
        return false;
    if (maxval != 255u || width == 0u || height == 0u)
        return false;

    // Exactly one whitespace byte separates the header from the pixel data.
    if (c.pos >= length || !isSpace(data[c.pos]))
        return false;
    ++c.pos;

    const core::usize pixels = static_cast<core::usize>(width) * height;
    if (c.pos + pixels * 3u > length)
        return false;

    out.resize(width, height);
    for (core::u32 y = 0u; y < height; ++y)
        for (core::u32 x = 0u; x < width; ++x)
        {
            const core::u8 r = data[c.pos++];
            const core::u8 g = data[c.pos++];
            const core::u8 b = data[c.pos++];
            out.set(static_cast<core::i32>(x), static_cast<core::i32>(y), packRgba(r, g, b));
        }
    return true;
}

} // namespace lpl::image
