/**
 * @file Extrusion.hpp
 * @brief Turning a 2D plan into volume, without paying for 3D generation.
 *
 * Running WFC in three dimensions is where the approach stops being practical:
 * a cube module has up to 24 rotations and symmetries, adjacency rules multiply
 * across six faces instead of four, and the odds of painting the solver into a
 * dead end climb with every added cell. The industry answer is not a better 3D
 * solver — it is to solve strictly in 2D and give the result height afterwards.
 *
 * That is what this does. A tile grid (from WFC, a dungeon, a settlement) plus a
 * height rule becomes a column per cell, and the result is a voxel volume that
 * cost a 2D solve. The pattern is often called 2.5D for that reason: full
 * volumetric output, planar generation.
 *
 * Heights are quantised to whole levels on purpose. A continuous height per cell
 * would leave every column a different size and no two neighbours able to share
 * a wall; snapping to integer steps is what lets sloped and stair pieces connect
 * exactly, which is the same hierarchical clamping a heightmap-driven chunk
 * system relies on.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_EXTRUSION_HPP
#    define LPL_PROCGEN_EXTRUSION_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/**
 * @struct VoxelVolume
 * @brief A dense 3D grid of material ids, stored column-major by level.
 *
 * Index order is (level, z, x) so a whole level is contiguous — which is the
 * order a mesher, a renderer or a collision builder walks it in.
 */
struct VoxelVolume {
    core::u32 width{0u};   ///< Cells along X.
    core::u32 depth{0u};   ///< Cells along Z.
    core::u32 levels{0u};  ///< Cells along Y.
    lpl::pmr::vector<core::u8> cells; ///< Material per voxel; 0 means empty.

    /// @return Flat index of (@p x, @p y, @p z).
    [[nodiscard]] core::u32 index(core::u32 x, core::u32 y, core::u32 z) const noexcept
    {
        return x + z * width + y * width * depth;
    }

    [[nodiscard]] core::u8 &at(core::u32 x, core::u32 y, core::u32 z) noexcept { return cells[index(x, y, z)]; }
    [[nodiscard]] core::u8 at(core::u32 x, core::u32 y, core::u32 z) const noexcept { return cells[index(x, y, z)]; }

    [[nodiscard]] bool empty() const noexcept { return width == 0u || depth == 0u || levels == 0u; }
    [[nodiscard]] core::u32 voxelCount() const noexcept { return width * depth * levels; }
};

/**
 * @struct ExtrusionParams
 * @brief How a plan gains height.
 */
struct ExtrusionParams {
    core::u32 levels{8u};        ///< Vertical extent of the volume.
    core::u32 baseLevel{0u};     ///< Level the ground starts at.
    core::f32 heightScale{0.25f};///< World units per level, when extruding a heightfield.
    bool solidBelow{true};       ///< Fill every level under the surface, not just the surface itself.
};

/**
 * @brief Extrudes a tile plan by a fixed height per tile id.
 *
 * The height rule is a lookup table: tile 3 is four levels tall, tile 0 is
 * empty, and so on. That keeps the plan and the volume in one relationship a
 * caller can read at a glance.
 *
 * @param plan       The 2D arrangement.
 * @param tileHeight Level count per tile id; entries beyond its size are 1.
 * @param params     Volume extent and fill mode.
 * @return The volume.
 */
[[nodiscard]] VoxelVolume extrudeTilePlan(const Grid<core::u8> &plan, const lpl::pmr::vector<core::u8> &tileHeight,
                                          const ExtrusionParams &params);

/**
 * @brief Extrudes a heightfield into voxels, quantising to whole levels.
 * @param field    Terrain to voxelise.
 * @param material Material id written for solid voxels.
 * @param params   Volume extent, scale and fill mode.
 * @return The volume.
 */
[[nodiscard]] VoxelVolume extrudeHeightfield(const Heightfield &field, core::u8 material,
                                             const ExtrusionParams &params);

/**
 * @brief Counts the voxels whose material is not 0.
 * @param volume Volume to measure.
 * @return Solid voxel count.
 */
[[nodiscard]] core::u32 countSolidVoxels(const VoxelVolume &volume);

/**
 * @brief Counts solid voxels with at least one empty face — the visible shell.
 *
 * What a mesher would actually emit. Useful as a cheap sanity check that an
 * extrusion produced a surface rather than a solid block.
 *
 * @param volume Volume to measure.
 * @return Surface voxel count.
 */
[[nodiscard]] core::u32 countSurfaceVoxels(const VoxelVolume &volume);

/**
 * @brief FNV-1a fold of a volume, for determinism checks.
 * @param volume Volume to fold.
 * @return The 32-bit signature.
 */
[[nodiscard]] core::u32 foldVolume(const VoxelVolume &volume);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_EXTRUSION_HPP
