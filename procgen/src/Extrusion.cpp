/**
 * @file Extrusion.cpp
 * @brief Implementation of 2.5D extrusion from plans and heightfields.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Extrusion.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/// Allocates an empty volume of the requested extent.
VoxelVolume makeVolume(core::u32 width, core::u32 depth, core::u32 levels)
{
    VoxelVolume volume;
    if (width == 0u || depth == 0u || levels == 0u)
        return volume;
    volume.width = width;
    volume.depth = depth;
    volume.levels = levels;
    volume.cells.resize(static_cast<core::usize>(width) * depth * levels, core::u8{0});
    return volume;
}

/// Writes a column from @p base up to (but excluding) @p top.
void fillColumn(VoxelVolume &volume, core::u32 x, core::u32 z, core::u32 base, core::u32 top, core::u8 material,
                bool solidBelow)
{
    if (top <= base)
        return;
    const core::u32 ceiling = top > volume.levels ? volume.levels : top;

    if (solidBelow)
    {
        for (core::u32 y = base; y < ceiling; ++y)
            volume.at(x, y, z) = material;
        return;
    }

    // Surface only: one voxel at the top of the column. A hollow world is far
    // cheaper to mesh, and nothing under the surface is ever seen.
    if (ceiling > base)
        volume.at(x, ceiling - 1u, z) = material;
}

} // namespace

VoxelVolume extrudeTilePlan(const Grid<core::u8> &plan, const lpl::pmr::vector<core::u8> &tileHeight,
                            const ExtrusionParams &params)
{
    VoxelVolume volume = makeVolume(plan.width(), plan.depth(), params.levels);
    if (volume.empty())
        return volume;

    for (core::u32 z = 0u; z < plan.depth(); ++z)
    {
        for (core::u32 x = 0u; x < plan.width(); ++x)
        {
            const core::u8 tile = plan.at(x, z);
            const core::u32 height =
                tile < tileHeight.size() ? static_cast<core::u32>(tileHeight[tile]) : 1u;
            if (height == 0u)
                continue; // this tile is a hole, not a column

            // Material carries the tile id, so the volume still knows what each
            // column was made of — a mesher can texture from it without needing
            // the plan alongside.
            fillColumn(volume, x, z, params.baseLevel, params.baseLevel + height,
                       static_cast<core::u8>(tile + 1u), params.solidBelow);
        }
    }
    return volume;
}

VoxelVolume extrudeHeightfield(const Heightfield &field, core::u8 material, const ExtrusionParams &params)
{
    VoxelVolume volume = makeVolume(field.width(), field.depth(), params.levels);
    if (volume.empty())
        return volume;

    const math::Fixed32 scale = math::Fixed32::fromFloat(params.heightScale <= 0.0f ? 1.0f : params.heightScale);

    for (core::u32 z = 0u; z < field.depth(); ++z)
    {
        for (core::u32 x = 0u; x < field.width(); ++x)
        {
            // Quantise to whole levels: neighbouring columns must share exact
            // wall heights or nothing connects.
            const core::i32 levels = (field.at(x, z) / scale).toInt();
            if (levels <= 0)
                continue;

            const core::u32 top = params.baseLevel + static_cast<core::u32>(levels);
            fillColumn(volume, x, z, params.baseLevel, top, material, params.solidBelow);
        }
    }
    return volume;
}

core::u32 countSolidVoxels(const VoxelVolume &volume)
{
    core::u32 count = 0u;
    for (core::u32 i = 0u; i < volume.cells.size(); ++i)
        if (volume.cells[i] != 0u)
            ++count;
    return count;
}

core::u32 countSurfaceVoxels(const VoxelVolume &volume)
{
    if (volume.empty())
        return 0u;

    core::u32 count = 0u;
    for (core::u32 y = 0u; y < volume.levels; ++y)
    {
        for (core::u32 z = 0u; z < volume.depth; ++z)
        {
            for (core::u32 x = 0u; x < volume.width; ++x)
            {
                if (volume.at(x, y, z) == 0u)
                    continue;

                // Outside the volume counts as empty, so the outer shell is
                // surface — which is what a mesher emits too.
                bool exposed = x == 0u || z == 0u || y == 0u || x + 1u == volume.width ||
                               z + 1u == volume.depth || y + 1u == volume.levels;
                if (!exposed)
                    exposed = volume.at(x + 1u, y, z) == 0u || volume.at(x - 1u, y, z) == 0u ||
                              volume.at(x, y + 1u, z) == 0u || volume.at(x, y - 1u, z) == 0u ||
                              volume.at(x, y, z + 1u) == 0u || volume.at(x, y, z - 1u) == 0u;
                if (exposed)
                    ++count;
            }
        }
    }
    return count;
}

core::u32 foldVolume(const VoxelVolume &volume)
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < volume.cells.size(); ++i)
        hash = (hash ^ static_cast<core::u32>(volume.cells[i])) * kFnv1aPrime;
    return hash;
}

} // namespace lpl::procgen
