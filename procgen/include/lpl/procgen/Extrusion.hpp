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
    core::u32 width{0u};              ///< Cells along X.
    core::u32 depth{0u};              ///< Cells along Z.
    core::u32 levels{0u};             ///< Cells along Y.
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
    core::u32 levels{8u};         ///< Vertical extent of the volume.
    core::u32 baseLevel{0u};      ///< Level the ground starts at.
    core::f32 heightScale{0.25f}; ///< World units per level, when extruding a heightfield.
    bool solidBelow{true};        ///< Fill every level under the surface, not just the surface itself.
};

/**
 * @struct VoxelWindow
 * @brief The part of a volume a walk should look at. Default is all of it.
 *
 * Exists because the second consumer of the face walk draws into a software rasterizer
 * on a kernel budget, and a warren is ten thousand voxels of which a body underground
 * can see a few hundred. Clamped by the walk rather than trusted, so a caller that
 * computes a window from a camera position cannot walk off the end of the array.
 */
struct VoxelWindow {
    core::u32 minX{0u};
    core::u32 minY{0u};
    core::u32 minZ{0u};
    core::u32 maxX{0xFFFFFFFFu}; ///< Exclusive.
    core::u32 maxY{0xFFFFFFFFu};
    core::u32 maxZ{0xFFFFFFFFu};
};

/**
 * @brief Hands each exposed face of a volume to @p emit, with its outward normal.
 *
 * The walk, with no opinion about what a face becomes. It was written once to append
 * to a vertex buffer, and then a second consumer arrived that has no vertex buffer —
 * a software rasterizer filling polygons one at a time — so the choice was to write
 * the walk again or to separate it from its sink. This repository already knows what
 * writing it again costs: @ref appendVoxelFaces' own comment counts five copies of
 * "emit a quad per exposed face" and says every copy is a chance to wind one the wrong
 * way round.
 *
 * A face is emitted where a solid voxel touches a non-solid one, and NOT at the
 * volume's border in the window's sense: whether the outside of the array is solid is
 * the @p solidOutside predicate's business, because a building floats in air and a
 * cave is surrounded by rock.
 *
 * @param volume       What to walk.
 * @param window       Which part; clamped to the volume.
 * @param solidOutside Called for coordinates outside the ARRAY: (x, y, z) -> bool.
 * @param emit         `emit(const core::f32 quad[12], f32 nx, f32 ny, f32 nz, u8 material,
 *                      u32 x, u32 y, u32 z)`, in volume-local units where one voxel is
 *                      one unit and level 0 spans y in [0, 1).
 */
template <typename SolidOutside, typename Emit>
void forEachVoxelFace(const VoxelVolume &volume, const VoxelWindow &window, SolidOutside &&solidOutside, Emit &&emit)
{
    if (volume.empty())
        return;

    const core::u32 highX = window.maxX < volume.width ? window.maxX : volume.width;
    const core::u32 highY = window.maxY < volume.levels ? window.maxY : volume.levels;
    const core::u32 highZ = window.maxZ < volume.depth ? window.maxZ : volume.depth;

    const auto solid = [&volume, &solidOutside](core::i32 x, core::i32 y, core::i32 z) {
        if (x < 0 || y < 0 || z < 0 || static_cast<core::u32>(x) >= volume.width ||
            static_cast<core::u32>(y) >= volume.levels || static_cast<core::u32>(z) >= volume.depth)
            return static_cast<bool>(solidOutside(x, y, z));
        return volume.at(static_cast<core::u32>(x), static_cast<core::u32>(y), static_cast<core::u32>(z)) != 0u;
    };

    for (core::u32 y = window.minY; y < highY; ++y)
        for (core::u32 z = window.minZ; z < highZ; ++z)
            for (core::u32 x = window.minX; x < highX; ++x)
            {
                const core::u8 material = volume.at(x, y, z);
                if (material == 0u)
                    continue;

                const core::f32 x0 = static_cast<core::f32>(x);
                const core::f32 x1 = x0 + 1.0f;
                const core::f32 y0 = static_cast<core::f32>(y);
                const core::f32 y1 = y0 + 1.0f;
                const core::f32 z0 = static_cast<core::f32>(z);
                const core::f32 z1 = z0 + 1.0f;

                const core::i32 ix = static_cast<core::i32>(x);
                const core::i32 iy = static_cast<core::i32>(y);
                const core::i32 iz = static_cast<core::i32>(z);

                if (!solid(ix, iy + 1, iz))
                {
                    const core::f32 quad[12] = {x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1};
                    emit(quad, 0.0f, 1.0f, 0.0f, material, x, y, z);
                }
                if (!solid(ix, iy - 1, iz))
                {
                    const core::f32 quad[12] = {x0, y0, z1, x1, y0, z1, x1, y0, z0, x0, y0, z0};
                    emit(quad, 0.0f, -1.0f, 0.0f, material, x, y, z);
                }
                if (!solid(ix + 1, iy, iz))
                {
                    const core::f32 quad[12] = {x1, y0, z0, x1, y1, z0, x1, y1, z1, x1, y0, z1};
                    emit(quad, 1.0f, 0.0f, 0.0f, material, x, y, z);
                }
                if (!solid(ix - 1, iy, iz))
                {
                    const core::f32 quad[12] = {x0, y0, z1, x0, y1, z1, x0, y1, z0, x0, y0, z0};
                    emit(quad, -1.0f, 0.0f, 0.0f, material, x, y, z);
                }
                if (!solid(ix, iy, iz + 1))
                {
                    const core::f32 quad[12] = {x0, y0, z1, x0, y1, z1, x1, y1, z1, x1, y0, z1};
                    emit(quad, 0.0f, 0.0f, 1.0f, material, x, y, z);
                }
                if (!solid(ix, iy, iz - 1))
                {
                    const core::f32 quad[12] = {x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z0};
                    emit(quad, 0.0f, 0.0f, -1.0f, material, x, y, z);
                }
            }
}

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
