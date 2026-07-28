/**
 * @file Grid.hpp
 * @brief The substrate every procedural pass reads and writes.
 *
 * Generation used to go straight from noise to entities, which meant nothing
 * could be *changed* afterwards: you cannot erode a cube that is already an ECS
 * entity, nor carve a river through it, nor ask what biome it belongs to. So
 * every pass now operates on a grid, and entities are materialised once at the
 * end — the "points + attributes, then realise" model the UE5 PCG study
 * describes.
 *
 * One template serves the three grids the passes need (heights, tiles, biomes),
 * because they differ only in cell type. Storage is a single flat
 * `lpl::pmr::vector` in row-major order, so a grid is kernel-safe and its
 * iteration order — hence every fold computed from it — is fixed.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_GRID_HPP
#    define LPL_PROCGEN_GRID_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/**
 * @class Grid
 * @brief A dense row-major 2D array of @p T, sized once.
 *
 * @tparam T Cell type; must be default-constructible and trivially copyable.
 */
template <typename T> class Grid {
public:
    Grid() = default;

    /**
     * @brief Allocates a @p width x @p depth grid filled with @p initial.
     * @param width   Cells along X.
     * @param depth   Cells along Z.
     * @param initial Value every cell starts at.
     */
    Grid(core::u32 width, core::u32 depth, T initial = T{})
        : _width(width), _depth(depth), _cells(static_cast<core::usize>(width) * depth, initial)
    {
    }

    [[nodiscard]] core::u32 width() const noexcept { return _width; }
    [[nodiscard]] core::u32 depth() const noexcept { return _depth; }
    [[nodiscard]] core::u32 cellCount() const noexcept { return _width * _depth; }
    [[nodiscard]] bool empty() const noexcept { return _width == 0u || _depth == 0u; }

    /// @return true when (@p x, @p z) is inside the grid.
    [[nodiscard]] bool contains(core::i32 x, core::i32 z) const noexcept
    {
        return x >= 0 && z >= 0 && static_cast<core::u32>(x) < _width && static_cast<core::u32>(z) < _depth;
    }

    /// @return Row-major index of (@p x, @p z). Caller must have checked bounds.
    [[nodiscard]] core::u32 index(core::u32 x, core::u32 z) const noexcept { return x + z * _width; }

    [[nodiscard]] T &at(core::u32 x, core::u32 z) noexcept { return _cells[index(x, z)]; }
    [[nodiscard]] const T &at(core::u32 x, core::u32 z) const noexcept { return _cells[index(x, z)]; }

    [[nodiscard]] T &operator[](core::u32 flatIndex) noexcept { return _cells[flatIndex]; }
    [[nodiscard]] const T &operator[](core::u32 flatIndex) const noexcept { return _cells[flatIndex]; }

    /**
     * @brief Reads a cell, clamping out-of-range coordinates to the edge.
     *
     * Every pass has to decide what lies beyond the border. Clamping (rather
     * than wrapping or returning a constant) keeps slopes finite at the edges,
     * so erosion does not carve a cliff into the boundary and rivers do not
     * flow off into a fictitious hole.
     *
     * @param x Column, may be out of range.
     * @param z Row, may be out of range.
     * @return The nearest in-range cell.
     */
    [[nodiscard]] const T &clamped(core::i32 x, core::i32 z) const noexcept
    {
        const core::u32 cx = x < 0 ? 0u : (static_cast<core::u32>(x) >= _width ? _width - 1u : static_cast<core::u32>(x));
        const core::u32 cz = z < 0 ? 0u : (static_cast<core::u32>(z) >= _depth ? _depth - 1u : static_cast<core::u32>(z));
        return _cells[index(cx, cz)];
    }

    /// @brief Sets every cell to @p value.
    void fill(T value) noexcept
    {
        for (core::u32 i = 0u; i < cellCount(); ++i)
            _cells[i] = value;
    }

    /// @return Pointer to the flat cell array (row-major), or nullptr when empty.
    [[nodiscard]] T *data() noexcept { return _cells.empty() ? nullptr : &_cells[0]; }
    [[nodiscard]] const T *data() const noexcept { return _cells.empty() ? nullptr : &_cells[0]; }

private:
    core::u32 _width{0u};
    core::u32 _depth{0u};
    lpl::pmr::vector<T> _cells;
};

/// The four axis-aligned neighbour offsets, in a fixed order (E, W, S, N).
inline constexpr core::i32 kNeighbor4X[4] = {1, -1, 0, 0};
inline constexpr core::i32 kNeighbor4Z[4] = {0, 0, 1, -1};

/// The eight neighbour offsets, in a fixed order. Order matters: it decides
/// which of several equal candidates a pass picks, and therefore the result.
inline constexpr core::i32 kNeighbor8X[8] = {1, -1, 0, 0, 1, 1, -1, -1};
inline constexpr core::i32 kNeighbor8Z[8] = {0, 0, 1, -1, 1, -1, 1, -1};

} // namespace lpl::procgen

#endif // LPL_PROCGEN_GRID_HPP
