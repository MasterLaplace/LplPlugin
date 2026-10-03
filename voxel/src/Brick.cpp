/**
 * @file Brick.cpp
 * @brief Summarising a brick so rays can skip it without reading it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voxel/Brick.hpp>

namespace lpl::voxel {

void summarise(BrickView &brick) noexcept
{
    if (!brick.valid())
    {
        brick.lowest = 255u;
        brick.highest = 0u;
        return;
    }

    core::u8 lo = 255u;
    core::u8 hi = 0u;
    for (core::u32 i = 0u; i < kBrickVoxels; ++i)
    {
        const core::u8 v = brick.voxels[i];
        if (v < lo)
            lo = v;
        if (v > hi)
            hi = v;
    }
    brick.lowest = lo;
    brick.highest = hi;
}

void summariseCells(BrickView &brick, core::u8 *out) noexcept
{
    summarise(brick);
    brick.occupancy = nullptr;
    if (!brick.valid() || out == nullptr)
        return;

    for (core::u32 i = 0u; i < kOccupancyEdge * kOccupancyEdge * kOccupancyEdge; ++i)
    {
        out[i * 2u] = 255u;
        out[i * 2u + 1u] = 0u;
    }
    for (core::u32 lz = 0u; lz < kBrickEdge; ++lz)
    {
        for (core::u32 ly = 0u; ly < kBrickEdge; ++ly)
        {
            const core::u32 rowCell =
                ((lz >> kOccupancyShift) * kOccupancyEdge + (ly >> kOccupancyShift)) * kOccupancyEdge;
            const core::u8 *row = &brick.voxels[(static_cast<core::usize>(lz) * kBrickEdge + ly) * kBrickEdge];
            for (core::u32 lx = 0u; lx < kBrickEdge; ++lx)
            {
                const core::u32 cell = (rowCell + (lx >> kOccupancyShift)) * 2u;
                const core::u8 v = row[lx];
                if (v < out[cell])
                    out[cell] = v;
                if (v > out[cell + 1u])
                    out[cell + 1u] = v;
            }
        }
    }
    brick.occupancy = out;
}

} // namespace lpl::voxel
