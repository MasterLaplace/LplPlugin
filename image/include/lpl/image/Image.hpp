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

    /**
     * @brief Resize to @p width x @p height, clearing to transparent black.
     *
     * clear()+resize() (rather than assign()) keeps this portable to the
     * freestanding kstd::vector, which has no assign().
     */
    void resize(core::u32 width, core::u32 height);

    [[nodiscard]] core::u32 width() const noexcept { return _width; }
    [[nodiscard]] core::u32 height() const noexcept { return _height; }
    [[nodiscard]] bool empty() const noexcept { return _pixels.empty(); }
    [[nodiscard]] const Rgba *data() const noexcept { return _pixels.data(); }
    [[nodiscard]] Rgba *data() noexcept { return _pixels.data(); }

    /**
     * @brief Fill the whole image with @p color.
     */
    void fill(Rgba color);

    /**
     * @brief Read a pixel with coordinates clamped to the image bounds.
     */
    [[nodiscard]] Rgba at(core::i32 x, core::i32 y) const noexcept;

    /**
     * @brief Write a pixel; out-of-bounds writes are ignored.
     */
    void set(core::i32 x, core::i32 y, Rgba color) noexcept;

    /**
     * @brief Nearest-neighbour sample at normalized Q16 coordinates.
     * @param u Horizontal coordinate in Q16 (0x10000 == right edge).
     * @param v Vertical coordinate in Q16 (0x10000 == bottom edge).
     */
    [[nodiscard]] Rgba sampleNearest(core::u32 u, core::u32 v) const noexcept;

    /**
     * @brief Bilinear sample at normalized Q16 coordinates (integer-weighted).
     *
     * Maps normalized Q16 to a Q16 pixel coordinate biased by -0.5 px so samples
     * land at texel centres, then splits it into integer index + 0..256 weight.
     */
    [[nodiscard]] Rgba sampleBilinear(core::u32 u, core::u32 v) const noexcept;

    /**
     * @brief Accumulate per-channel and luminance histograms over all pixels.
     */
    [[nodiscard]] Histogram histogram() const noexcept;

private:
    [[nodiscard]] static core::u32 clampCoord(core::i32 v, core::u32 size) noexcept;

    core::u32 _width = 0u;
    core::u32 _height = 0u;
    pmr::vector<Rgba> _pixels;
};

} // namespace lpl::image

#    include "Image.inl"

#endif // LPL_IMAGE_IMAGE_HPP
