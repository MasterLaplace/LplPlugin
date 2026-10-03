/**
 * @file Minimap.hpp
 * @brief Where you are in the whole thing, drawn in a corner of the frame you are already making.
 *
 * @warning **A dot on a box would not answer the question somebody actually has.** Inside a
 * scanned object every direction looks like every other direction, and what a person is lost about
 * is not their coordinates -- it is which part of the STRUCTURE they are in. So the panel draws a
 * real slice of the subject with the eye on it: on a rolled scroll that is the spiral, and one
 * glance says which turn you are between and how far from the axis.
 *
 * @warning **The slice is meant to come from the coarsest pyramid level, and that is what makes it
 * free.** The whole of PHerc0172 at 253 micrometres is thirty-seven megabytes -- resident
 * permanently, no streaming, no waiting -- while its finest level is 1.2 terabytes. A minimap fed
 * from the same bricks the walk uses would only ever show the few metres already on screen, which
 * is the one thing a lost person does not need. The caller passes the mosaic that holds that level:
 * see @ref extractSlice.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOXEL_MINIMAP_HPP
#    define LPL_VOXEL_MINIMAP_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/voxel/Mosaic.hpp>
#    include <lpl/voxel/Raymarch.hpp>
#    include <lpl/voxel/Volume.hpp>

namespace lpl::voxel {

/**
 * @enum SliceAxis
 * @brief Which volume axis the slice is perpendicular to, and so which two it spans.
 *
 * | axis | the column index steps along | the row index steps along |
 * |------|------------------------------|---------------------------|
 * | Z    | x                            | y                         |
 * | Y    | x                            | z                         |
 * | X    | y                            | z                         |
 *
 * A value outside these three is read as X.
 */
enum class SliceAxis : core::u8 {
    Z = 0, ///< Across the volume's slowest axis. On a rolled scroll, the spiral in cross-section.
    Y,
    X,
};

/**
 * @struct VolumeSlice
 * @brief A plane of samples cut out of a volume, in caller-owned storage.
 *
 * Column c and row r hold the level-0 sample at `lowU + c * stepU` along the columns' axis and
 * `lowV + r * stepV` along the rows' axis (@ref SliceAxis). @ref extractSlice writes that mapping
 * and @ref drawMinimap places the eye through it, so the picture and the marker put a sample at
 * the same spot.
 */
struct VolumeSlice final {
    core::u8 *samples{nullptr}; ///< @ref width * @ref height samples, row after row. 0 where none was read.
    core::u32 width{0};         ///< Columns.
    core::u32 height{0};        ///< Rows.
    SliceAxis axis{SliceAxis::Z};
    core::i64 at{0};    ///< Level-0 sample along @ref axis where the plane is cut.
    core::i64 lowU{0};  ///< Level-0 sample at column zero.
    core::i64 lowV{0};  ///< Level-0 sample at row zero.
    core::i64 stepU{1}; ///< Level-0 samples per column.
    core::i64 stepV{1}; ///< Level-0 samples per row.

    [[nodiscard]] constexpr bool valid() const noexcept { return samples != nullptr && width != 0u && height != 0u; }
};

/**
 * @brief Fills @p slice with the samples of @p mosaic on the plane @ref VolumeSlice::at across
 * @ref VolumeSlice::axis.
 *
 * The slice always spans the entire volume, whatever the resolution asked for: a map that showed
 * only part of the subject would be a second view of where you already are. Each step is the
 * smallest whose columns (or rows) reach across the volume, so when the volume's extent is not a
 * multiple of the slice's width (or height), the last columns (or rows) lie past it and read 0.
 *
 * Reads the storage, width, height, axis and at of @p slice. Writes its samples, its lowU, lowV,
 * stepU and stepV, and its at, clamped into the volume so it says where the plane was cut.
 *
 * @pre @p mosaic holds the coarsest pyramid level and nothing finer. Each sample is read from the
 * finest resident brick that covers it (@ref BrickMosaic::sampleAt), so where a finer brick is
 * resident it answers instead, and the slice becomes a patchwork of resolutions that shows only
 * what happens to be loaded.
 * @return Samples read from a resident brick: 0 when no brick covering the plane is resident, and
 * 0 when @p slice or @p geometry is not valid, in which case nothing is written.
 */
[[nodiscard]] core::u32 extractSlice(const BrickMosaic &mosaic, const VolumeGeometry &geometry,
                                     VolumeSlice &slice) noexcept;

/**
 * @struct MinimapStyle
 * @brief Where the panel goes in the frame, and what it looks like. Colours are 0xAARRGGBB.
 */
struct MinimapStyle final {
    core::u32 left{0}; ///< Panel position in the frame, in pixels from its top-left corner.
    core::u32 top{0};
    core::u32 width{200}; ///< Panel size in pixels.
    core::u32 height{200};
    core::u32 border{0xFFE0912Cu};
    core::u32 background{0xFF0D0F12u}; ///< Where the slice holds 0.
    core::u32 marker{0xFF5BD6FFu};     ///< The eye and its heading.
    core::f32 sliceBrightness{0.75f};  ///< Grey of the slice's brightest sample, as a fraction of white clamped to
                                       ///< [0, 1]. Below 1, the marker stays readable on it.
    core::f32 headingLength{22.0f};    ///< Pixels of the line that shows the heading.
};

/**
 * @brief Draws the slice, a frame around it, and where the eye is, into the panel @p style places
 * in @p pixels.
 *
 * The slice is stretched over the panel in its own contrast: its darkest non-zero sample is black,
 * its brightest is @ref MinimapStyle::sliceBrightness of white, and a sample of 0 is the
 * background. The frame is the panel's outermost pixels.
 *
 * The eye is a dot at its place in the slice's columns and rows, clamped onto the panel when it is
 * outside the volume: you can fly out of a scroll, and a marker that vanished there would read as
 * the map being broken. Its heading is a line along its projection into the slice, or a ring around
 * the dot when it points within about thirteen degrees of the slice's axis: on a scroll the
 * interesting direction is down the axis, which is the one the slice is cut across, so a line
 * would vanish exactly when somebody is doing the normal thing. The eye's depth along that axis is
 * not compared with @ref VolumeSlice::at.
 *
 * @warning **The marker is drawn from the same geometry the march uses**, not from a position the
 * caller passed separately. Two answers to "where is the eye" is how a map ends up confidently
 * pointing at the wrong turn of a spiral, and looking entirely plausible doing it.
 *
 * @param pixels  The frame, @p frameWidth * @p frameHeight entries, 0xAARRGGBB, row after row. Only
 *                the panel's pixels are written.
 * @pre @p slice was filled by @ref extractSlice with @p geometry.
 * @return Whether the panel was drawn. Nothing is drawn when @p pixels is null, when @p slice or
 * @p geometry is not valid, when a step of @p slice is below 1 or the metres per sample of
 * @p geometry rounds to zero, or when the panel is empty or does not fit inside the frame.
 */
[[nodiscard]] bool drawMinimap(core::u32 *pixels, core::u32 frameWidth, core::u32 frameHeight,
                               const MinimapStyle &style, const VolumeSlice &slice, const VolumeGeometry &geometry,
                               const Eye &eye) noexcept;

} // namespace lpl::voxel

#endif // LPL_VOXEL_MINIMAP_HPP
