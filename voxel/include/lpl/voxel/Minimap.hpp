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
 * @warning **The slice comes from the coarsest pyramid level, and that is what makes it free.**
 * The whole of PHerc0172 at 253 micrometres is thirty-seven megabytes -- resident permanently, no
 * streaming, no waiting -- while its finest level is 1.2 terabytes. A minimap fed from the same
 * bricks the walk uses would only ever show the few metres already on screen, which is the one
 * thing a lost person does not need.
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
 * @brief Which axis the slice is perpendicular to.
 */
enum class SliceAxis : core::u8 {
    Z = 0, ///< Across the volume's slowest axis. On a rolled scroll, the spiral in cross-section.
    Y,
    X,
};

/**
 * @struct VolumeSlice
 * @brief A plane of samples cut out of a volume, in caller-owned storage.
 */
struct VolumeSlice final {
    core::u8 *samples{nullptr};
    core::u32 width{0}; ///< Along the first axis that is not @ref axis.
    core::u32 height{0};
    SliceAxis axis{SliceAxis::Z};
    core::i64 at{0};    ///< Position along @ref axis, in level-0 samples.
    core::i64 lowU{0};  ///< Level-0 sample at column zero.
    core::i64 lowV{0};  ///< Level-0 sample at row zero.
    core::i64 stepU{1}; ///< Level-0 samples per column.
    core::i64 stepV{1}; ///< Level-0 samples per row.

    [[nodiscard]] constexpr bool valid() const noexcept { return samples != nullptr && width != 0u && height != 0u; }
};

/**
 * @brief Fills @p slice by sampling @p mosaic across the whole subject.
 *
 * The slice always spans the entire volume, whatever the resolution asked for: a map that showed
 * only part of the subject would be a second view of where you already are.
 *
 * @return Samples that found a resident brick. Zero means the atlas is not loaded.
 */
core::u32 extractSlice(const BrickMosaic &mosaic, const VolumeGeometry &geometry, VolumeSlice &slice) noexcept;

/**
 * @struct MinimapStyle
 * @brief Where the panel goes and what it looks like.
 */
struct MinimapStyle final {
    core::u32 left{0};
    core::u32 top{0};
    core::u32 width{200};
    core::u32 height{200};
    core::u32 border{0xFFE0912Cu}; ///< 0xAARRGGBB.
    core::u32 background{0xFF0D0F12u};
    core::u32 marker{0xFF5BD6FFu};
    core::f32 dim{0.75f};        ///< How far the slice is darkened, so the marker stays readable.
    core::f32 coneLength{22.0f}; ///< Pixels of the heading indicator.
};

/**
 * @brief Draws the slice, a frame around it, and where the eye is, into @p pixels.
 *
 * @warning **The marker is drawn from the same geometry the march uses**, not from a position the
 * caller passed separately. Two answers to "where is the eye" is how a map ends up confidently
 * pointing at the wrong turn of a spiral, and looking entirely plausible doing it.
 */
void drawMinimap(core::u32 *pixels, core::u32 width, core::u32 height, const MinimapStyle &style,
                 const VolumeSlice &slice, const VolumeGeometry &geometry, const Eye &eye) noexcept;

} // namespace lpl::voxel

#endif // LPL_VOXEL_MINIMAP_HPP
