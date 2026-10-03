/**
 * @file Brick.hpp
 * @brief One cube of scanned matter, and where it sits in the scroll.
 *
 * A brick is 128 x 128 x 128 single-byte samples, which is not a number this module chose: it is
 * the chunk shape the published OME-Zarr volumes use, so a brick is exactly one HTTP request, one
 * cache record and one unit of residency. Making it anything else would mean cutting or gluing on
 * every load, forever, for no gain.
 *
 * @warning **A brick is a VIEW, never storage.** The bytes live in a memory-mapped cache record,
 * and the whole point of mapping is that they are never copied. A brick that owned its bytes would
 * turn a 2 MiB page-cache reference into a 2 MiB allocation, times the resident set, and the
 * resident set is the budget the whole design is built around.
 *
 * @warning **The summary is computed once, on load, and it is what makes empty space cheap.** A ray
 * that can ask "is there anything in this brick above the first visible density" without touching
 * two million bytes skips it in one comparison. Recomputing it per ray would cost more than not
 * having it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOXEL_BRICK_HPP
#    define LPL_VOXEL_BRICK_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::voxel {

/// Samples along one brick edge. The published chunk shape; see the file header.
inline constexpr core::u32 kBrickEdge = 128u;

/// log2 of @ref kBrickEdge. Named rather than open-coded so the two cannot drift apart.
inline constexpr core::u32 kBrickEdgeShift = 7u;
static_assert((1u << kBrickEdgeShift) == kBrickEdge, "the shift and the edge must describe one number");

/// Samples in a whole brick.
inline constexpr core::u32 kBrickVoxels = kBrickEdge * kBrickEdge * kBrickEdge;

/// Levels the published pyramid carries: level 0 is finest, each step halves every axis.
inline constexpr core::u32 kMaxPyramidLevels = 8u;

/// Occupancy cells along one brick edge. Each covers @ref kBrickEdge / this samples.
inline constexpr core::u32 kOccupancyEdge = 8u;

/// log2 of the samples one occupancy cell spans.
inline constexpr core::u32 kOccupancyShift = 4u;
static_assert((kOccupancyEdge << kOccupancyShift) == kBrickEdge, "the cell grid must tile the brick");

/// Bytes an occupancy grid needs: one low and one high per cell.
inline constexpr core::u32 kOccupancyBytes = kOccupancyEdge * kOccupancyEdge * kOccupancyEdge * 2u;

/**
 * @struct BrickKey
 * @brief Which brick, at which resolution.
 *
 * @warning Coordinates are BRICK indices at their own level, and they are signed for the same
 * reason cell indices are elsewhere in this project: an index is arithmetic, and arithmetic that
 * cannot go negative turns a subtraction into a wraparound at the origin.
 */
struct BrickKey final {
    core::u32 level{0u}; ///< Pyramid level; 0 is finest.
    core::i32 z{0};      ///< Brick index along the scroll axis.
    core::i32 y{0};      ///< Brick index along Y.
    core::i32 x{0};      ///< Brick index along X.

    [[nodiscard]] constexpr bool operator==(const BrickKey &rhs) const noexcept
    {
        return level == rhs.level && z == rhs.z && y == rhs.y && x == rhs.x;
    }
};

/**
 * @struct BrickView
 * @brief A resident brick: the bytes, where they belong, and what is in them.
 */
struct BrickView final {
    const core::u8 *voxels{nullptr}; ///< kBrickVoxels bytes, C order (z, y, x). Not owned.
    BrickKey key{};                  ///< Which brick these bytes are.
    core::u8 lowest{255};            ///< Smallest sample in the brick.
    core::u8 highest{0};             ///< Largest sample in the brick.

    /**
     * Low and high per @ref kOccupancyEdge cubed cell, or null.
     *
     * @warning **This is what makes the medium cheap, and the medium is almost all of a scan.**
     * Measured: ninety-eight per cent of the samples a frame takes paint nothing -- they are in
     * the material between sheets, which is dark and not empty, so the whole-brick summary cannot
     * skip it. A cell that cannot reach the visible band is skipped in one comparison instead of
     * sixteen samples. It costs a kibibyte against a two-mebibyte brick: five hundredths of a per
     * cent.
     *
     * @warning It is EXACT on the samples, not a heuristic: a cell is skipped only when its highest
     * sample is below the first density the curve paints. An interpolated point near the cell's
     * face can still lean toward the next cell; @see MarchParams::skipEmptyCells for what that
     * costs.
     */
    const core::u8 *occupancy{nullptr};

    /**
     * @return Highest sample in the cell containing the given LOCAL coordinates.
     * @pre @ref occupancy is set, and each coordinate is below @ref kBrickEdge.
     */
    [[nodiscard]] constexpr core::u8 cellHighest(core::u32 lz, core::u32 ly, core::u32 lx) const noexcept
    {
        const core::u32 cell = ((lz >> kOccupancyShift) * kOccupancyEdge + (ly >> kOccupancyShift)) * kOccupancyEdge +
                               (lx >> kOccupancyShift);
        return occupancy[cell * 2u + 1u];
    }

    [[nodiscard]] constexpr bool valid() const noexcept { return voxels != nullptr; }

    /**
     * @return Sample at the given LOCAL coordinates.
     * @pre @ref valid(), and each coordinate is below @ref kBrickEdge.
     */
    [[nodiscard]] constexpr core::u8 at(core::u32 lz, core::u32 ly, core::u32 lx) const noexcept
    {
        return voxels[(static_cast<core::usize>(lz) * kBrickEdge + ly) * kBrickEdge + lx];
    }
};

/**
 * @brief Fills @ref BrickView::lowest and @ref BrickView::highest from the bytes.
 *
 * Call once when a brick becomes resident. Two million byte comparisons is a few milliseconds and
 * it buys every later ray the right to skip the brick without reading it.
 */
void summarise(BrickView &brick) noexcept;

/**
 * @brief Fills @ref BrickView::occupancy from the bytes, and points the view at @p out.
 *
 * @param out  At least @ref kOccupancyBytes, owned by the caller and outliving the view.
 */
void summariseCells(BrickView &brick, core::u8 *out) noexcept;

/**
 * @brief Edge of a brick expressed in level-0 samples.
 *
 * A level-L sample spans 2^L level-0 samples, so a level-L brick spans 128 * 2^L of them. This is
 * the only conversion between the pyramid and the world, and it lives here so nothing downstream
 * re-derives it.
 */
[[nodiscard]] constexpr core::i64 brickSpanInBaseSamples(core::u32 level) noexcept
{
    return static_cast<core::i64>(kBrickEdge) << level;
}

/**
 * @brief Lowest level-0 sample a brick covers on one axis.
 *
 * @param axis  0 for z, 1 for y, 2 for x: the zarr order of @ref BrickKey.
 * @pre @p axis < 3; any larger value reads x.
 */
[[nodiscard]] constexpr core::i64 brickOriginInBaseSamples(const BrickKey &key, core::u32 axis) noexcept
{
    const core::i32 index = axis == 0u ? key.z : (axis == 1u ? key.y : key.x);
    return static_cast<core::i64>(index) * brickSpanInBaseSamples(key.level);
}

/**
 * @brief Which brick of @p level contains the level-0 sample at @p base, on one axis.
 *
 * @warning Floor division, not truncation. C++ truncates toward zero, so `-1 / 128` is `0` and the
 * brick just below the origin would collide with the brick just above it. That is the same trap
 * this project already paid on a wrapped column index, and the fix is the same one.
 */
[[nodiscard]] constexpr core::i32 brickIndexOfBase(core::i64 base, core::u32 level) noexcept
{
    // An arithmetic shift rather than a division: a brick span is always a power of two, so the
    // shift IS the floor division, negatives included, where the old form paid a division and a
    // modulo. It matters because this is the innermost call of the renderer -- every sample makes
    // three and every gradient probe three more -- and the shift cut the frame by a third on real
    // data, with the picture unchanged to the bit.
    return static_cast<core::i32>(base >> (kBrickEdgeShift + level));
}

/**
 * @brief Floor of @p position as an i64: the level-0 sample that contains it, when it is given in
 * level-0 samples.
 *
 * @warning Floor, not truncation, for the reason @ref brickIndexOfBase gives: -0.5 is in sample -1.
 * @pre @p position is finite and within the range of an i64.
 */
template <typename Real> [[nodiscard]] constexpr core::i64 floorToI64(Real position) noexcept
{
    const core::i64 truncated = static_cast<core::i64>(position);
    return (position < Real{0} && static_cast<Real>(truncated) != position) ? truncated - 1 : truncated;
}

} // namespace lpl::voxel

#endif // LPL_VOXEL_BRICK_HPP
