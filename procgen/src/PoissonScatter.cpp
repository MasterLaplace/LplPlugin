/**
 * @file PoissonScatter.cpp
 * @brief Implementation of deterministic grid-jitter Poisson-disk scatter.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <lpl/procgen/PoissonScatter.hpp>

#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/procgen/ValueNoise.hpp>

#include <vector>

namespace lpl::procgen {

using FVec3 = math::Vec3<math::Fixed32>;

namespace {

// Ceil of a non-negative math::Fixed32 to an integer number of cells.
core::i32 ceilCells(math::Fixed32 v)
{
    return (v.raw() + 0xFFFF) >> 16;
}

} // namespace

core::u32 scatterPoisson(ecs::Registry &registry, const PoissonScatterParams &params)
{
    const math::Fixed32 width = math::Fixed32::fromFloat(params.width);
    const math::Fixed32 depth = math::Fixed32::fromFloat(params.depth);
    const math::Fixed32 radius = math::Fixed32::fromFloat(params.radius);
    if (width.raw() <= 0 || depth.raw() <= 0 || radius.raw() <= 0)
        return 0;

    // Background-grid cell size = radius / √2 so a filled cell holds one point and
    // any conflict lies within a 2-cell neighbourhood. √2⁻¹ is a math::Fixed32 constant.
    const math::Fixed32 invSqrt2 = math::Fixed32::fromFloat(0.70710678f);
    const math::Fixed32 cell = radius * invSqrt2;
    if (cell.raw() <= 0)
        return 0;

    const core::i32 gridW = ceilCells(width * (math::Fixed32::one() / cell));
    const core::i32 gridD = ceilCells(depth * (math::Fixed32::one() / cell));
    if (gridW <= 0 || gridD <= 0)
        return 0;

    const math::Fixed32 r2 = radius * radius;
    const math::Fixed32 halfW = width * math::Fixed32::half();
    const math::Fixed32 halfD = depth * math::Fixed32::half();

    // grid[gx + gy*gridW] = index into `points` of the accepted point in that
    // cell, or -1 if empty. Scan-order iteration keeps the result deterministic.
    std::vector<core::i32> grid(static_cast<std::size_t>(gridW) * static_cast<std::size_t>(gridD), -1);
    std::vector<FVec3> points;

    for (core::i32 gz = 0; gz < gridD; ++gz)
    {
        for (core::i32 gx = 0; gx < gridW; ++gx)
        {
            // Jittered candidate inside this cell (two hashed fractionals).
            const core::u32 ha = ValueNoise2D::hash2(gx, gz, params.seed);
            const core::u32 hb = ValueNoise2D::hash2(gx, gz, params.seed ^ 0x68BC21EBu);
            const math::Fixed32 jx = math::Fixed32::fromRaw(static_cast<core::i32>(ha & 0xFFFFu));
            const math::Fixed32 jz = math::Fixed32::fromRaw(static_cast<core::i32>(hb & 0xFFFFu));
            const math::Fixed32 px = (math::Fixed32::fromInt(gx) + jx) * cell - halfW;
            const math::Fixed32 pz = (math::Fixed32::fromInt(gz) + jz) * cell - halfD;

            // Reject if any accepted neighbour is closer than `radius`.
            bool ok = true;
            for (core::i32 nz = gz - 2; nz <= gz + 2 && ok; ++nz)
            {
                if (nz < 0 || nz >= gridD)
                    continue;
                for (core::i32 nx = gx - 2; nx <= gx + 2 && ok; ++nx)
                {
                    if (nx < 0 || nx >= gridW)
                        continue;
                    const core::i32 idx = grid[static_cast<std::size_t>(nx) + static_cast<std::size_t>(nz) * gridW];
                    if (idx < 0)
                        continue;
                    const FVec3 &q = points[static_cast<std::size_t>(idx)];
                    const math::Fixed32 dx = px - q.x;
                    const math::Fixed32 dz = pz - q.z;
                    if (dx * dx + dz * dz < r2)
                        ok = false;
                }
            }
            if (!ok)
                continue;
            if (params.maxPoints != 0u && points.size() >= params.maxPoints)
            {
                gz = gridD; // stop both loops
                break;
            }

            grid[static_cast<std::size_t>(gx) + static_cast<std::size_t>(gz) * gridW] =
                static_cast<core::i32>(points.size());
            points.push_back(FVec3{px, math::Fixed32::zero(), pz});
        }
    }

    const core::u32 total = static_cast<core::u32>(points.size());
    if (total == 0)
        return 0;

    const ecs::ComponentId ids[] = {ecs::ComponentId::Position, ecs::ComponentId::AABB};
    const ecs::Archetype archetype{ids};
    for (core::u32 i = 0; i < total; ++i)
        (void) registry.createEntity(archetype);

    const math::Fixed32 propSize = math::Fixed32::fromFloat(2.0f * params.propHalf);
    const FVec3 extents{propSize, propSize, propSize};

    core::u32 gi = 0;
    for (const auto &partition : registry.partitions())
    {
        if (!partition)
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            const core::u32 count = chunk->count();
            auto *pos = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *posR = static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Position)));
            auto *aabb = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::AABB));
            auto *aabbR = static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::AABB)));
            if (!pos)
                continue;
            for (core::u32 li = 0; li < count && gi < total; ++li, ++gi)
            {
                pos[li] = points[gi];
                if (posR)
                    posR[li] = points[gi];
                if (aabb)
                    aabb[li] = extents;
                if (aabbR)
                    aabbR[li] = extents;
            }
        }
    }
    return gi;
}

} // namespace lpl::procgen
