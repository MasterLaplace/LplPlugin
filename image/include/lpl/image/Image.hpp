/**
 * @file Image.hpp
 * @brief A simple RGBA8 image surface with deterministic sampling + histogram.
 *
 * Pixels are packed Rgba (0xAARRGGBB) stored row-major in a pmr::vector, so the
 * type compiles into libengine for the freestanding kernel build as well as the
 * Linux oracle. Sampling uses Q16 fixed-point UVs and integer-weighted blends —
 * no float — to stay bit-identical across targets.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_IMAGE_IMAGE_HPP
#    define LPL_IMAGE_IMAGE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/image/Color.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::image {

/** @brief Per-channel 256-bin histogram (plus a luminance histogram). */
struct Histogram {
    core::u32 red[256] = {};
    core::u32 green[256] = {};
    core::u32 blue[256] = {};
    core::u32 luminance[256] = {};
};

/**
 * @class Image
 * @brief Row-major RGBA8 surface with clamped access, sampling and histogram.
 */
class Image {
public:
    Image() = default;
    Image(core::u32 width, core::u32 height) { resize(width, height); }

    /** @brief Resize to @p width x @p height, clearing to transparent black. */
    void resize(core::u32 width, core::u32 height)
    {
        _width = width;
        _height = height;
        _pixels.assign(static_cast<core::usize>(width) * height, 0u);
    }

    [[nodiscard]] core::u32 width() const noexcept { return _width; }
    [[nodiscard]] core::u32 height() const noexcept { return _height; }
    [[nodiscard]] bool empty() const noexcept { return _pixels.empty(); }
    [[nodiscard]] const Rgba *data() const noexcept { return _pixels.data(); }
    [[nodiscard]] Rgba *data() noexcept { return _pixels.data(); }

    /** @brief Fill the whole image with @p color. */
    void fill(Rgba color)
    {
        for (auto &pixel : _pixels)
            pixel = color;
    }

    /** @brief Read a pixel with coordinates clamped to the image bounds. */
    [[nodiscard]] Rgba at(core::i32 x, core::i32 y) const noexcept
    {
        if (_width == 0u || _height == 0u)
            return 0u;
        const core::u32 cx = clampCoord(x, _width);
        const core::u32 cy = clampCoord(y, _height);
        return _pixels[static_cast<core::usize>(cy) * _width + cx];
    }

    /** @brief Write a pixel; out-of-bounds writes are ignored. */
    void set(core::i32 x, core::i32 y, Rgba color) noexcept
    {
        if (x < 0 || y < 0 || static_cast<core::u32>(x) >= _width || static_cast<core::u32>(y) >= _height)
            return;
        _pixels[static_cast<core::usize>(y) * _width + static_cast<core::u32>(x)] = color;
    }

    /**
     * @brief Nearest-neighbour sample at normalized Q16 coordinates.
     * @param u Horizontal coordinate in Q16 (0x10000 == right edge).
     * @param v Vertical coordinate in Q16 (0x10000 == bottom edge).
     */
    [[nodiscard]] Rgba sampleNearest(core::u32 u, core::u32 v) const noexcept
    {
        const core::i32 x = static_cast<core::i32>((static_cast<core::u64>(u) * _width) >> 16);
        const core::i32 y = static_cast<core::i32>((static_cast<core::u64>(v) * _height) >> 16);
        return at(x, y);
    }

    /** @brief Bilinear sample at normalized Q16 coordinates (integer-weighted). */
    [[nodiscard]] Rgba sampleBilinear(core::u32 u, core::u32 v) const noexcept;

    /** @brief Accumulate per-channel and luminance histograms over all pixels. */
    [[nodiscard]] Histogram histogram() const noexcept;

private:
    [[nodiscard]] static core::u32 clampCoord(core::i32 v, core::u32 size) noexcept
    {
        if (v < 0)
            return 0u;
        if (static_cast<core::u32>(v) >= size)
            return size - 1u;
        return static_cast<core::u32>(v);
    }

    core::u32 _width = 0u;
    core::u32 _height = 0u;
    pmr::vector<Rgba> _pixels;
};

} // namespace lpl::image

#endif // LPL_IMAGE_IMAGE_HPP
