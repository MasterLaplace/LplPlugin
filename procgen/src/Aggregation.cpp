/**
 * @file Aggregation.cpp
 * @brief Implementation of diffusion-limited aggregation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Aggregation.hpp>

#include <lpl/procgen/Random.hpp>

namespace lpl::procgen {

namespace {

/// Chebyshev distance, which matches the 8-neighbourhood the walk uses.
core::u32 chebyshev(core::i32 ax, core::i32 az, core::i32 bx, core::i32 bz)
{
    const core::u32 dx = static_cast<core::u32>(ax > bx ? ax - bx : bx - ax);
    const core::u32 dz = static_cast<core::u32>(az > bz ? az - bz : bz - az);
    return dx > dz ? dx : dz;
}

/**
 * @brief Is any 4-neighbour already part of the cluster?
 *
 * Four, not eight, even though the walk itself moves diagonally. Sticking on a
 * diagonal contact would produce a cluster that is 8-connected but NOT
 * 4-connected — and 4-connectivity is what isFullyConnected and every other
 * traversal in this module means by "connected". The repair pass would then dig
 * corridors through a structure that was already whole, quietly thickening the
 * dendrite it was supposed to leave alone.
 */
bool touchesCluster(const DungeonMap &map, core::i32 x, core::i32 z)
{
    for (core::u32 n = 0u; n < 4u; ++n)
    {
        const core::i32 nx = x + kNeighbor4X[n];
        const core::i32 nz = z + kNeighbor4Z[n];
        if (map.contains(nx, nz) && map.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) != DungeonCell::Wall)
            return true;
    }
    return false;
}

/// Carves a filled square of the given radius, clipped to the map.
void carveBlob(DungeonMap &map, core::i32 x, core::i32 z, core::u32 radius, core::u32 &outOpened)
{
    const core::i32 r = static_cast<core::i32>(radius);
    for (core::i32 dz = -r; dz <= r; ++dz)
    {
        for (core::i32 dx = -r; dx <= r; ++dx)
        {
            const core::i32 cx = x + dx;
            const core::i32 cz = z + dz;
            if (!map.contains(cx, cz))
                continue;
            DungeonCell &cell = map.at(static_cast<core::u32>(cx), static_cast<core::u32>(cz));
            if (cell == DungeonCell::Wall)
            {
                cell = DungeonCell::Floor;
                ++outOpened;
            }
        }
    }
}

} // namespace

DungeonMap generateDlaCave(const DlaParams &params, DlaReport *outReport)
{
    DlaReport report;
    if (params.width < 8u || params.depth < 8u)
    {
        if (outReport != nullptr)
            *outReport = report;
        return DungeonMap{};
    }

    DungeonMap map{params.width, params.depth, DungeonCell::Wall};
    Random random = deriveStream(params.seed, 0xD1Au);

    const core::i32 centerX = static_cast<core::i32>(params.width / 2u);
    const core::i32 centerZ = static_cast<core::i32>(params.depth / 2u);

    // The seed the cluster grows from.
    carveBlob(map, centerX, centerZ, params.thickness, report.openCells);

    // Walks are confined to a margin, and spawns track the cluster's current
    // extent: releasing from the far edge of a large map would spend nearly the
    // whole budget crossing empty rock before the cluster is even reachable.
    const core::u32 border = 1u;
    const core::u32 maxRadius = (params.width < params.depth ? params.width : params.depth) / 2u - border - 1u;

    for (core::u32 particle = 0u; particle < params.particles; ++particle)
    {
        core::u32 spawnRadius = report.extent + params.spawnMargin;
        if (spawnRadius > maxRadius)
            spawnRadius = maxRadius;
        if (spawnRadius < 2u)
            spawnRadius = 2u;

        // Spawn on the ring at spawnRadius: pick a side, then a position along it.
        const core::u32 side = random.below(4u);
        const core::i32 along = random.range(-static_cast<core::i32>(spawnRadius), static_cast<core::i32>(spawnRadius));
        core::i32 x = centerX;
        core::i32 z = centerZ;
        switch (side)
        {
        case 0u:
            x = centerX + static_cast<core::i32>(spawnRadius);
            z = centerZ + along;
            break;
        case 1u:
            x = centerX - static_cast<core::i32>(spawnRadius);
            z = centerZ + along;
            break;
        case 2u:
            x = centerX + along;
            z = centerZ + static_cast<core::i32>(spawnRadius);
            break;
        default:
            x = centerX + along;
            z = centerZ - static_cast<core::i32>(spawnRadius);
            break;
        }

        bool landed = false;
        for (core::u32 step = 0u; step < params.maxStepsPerParticle; ++step)
        {
            if (touchesCluster(map, x, z))
            {
                carveBlob(map, x, z, params.thickness, report.openCells);
                const core::u32 distance = chebyshev(x, z, centerX, centerZ);
                if (distance > report.extent)
                    report.extent = distance;
                ++report.stuck;
                landed = true;
                break;
            }

            const core::u32 direction = random.below(8u);
            const core::i32 nx = x + kNeighbor8X[direction];
            const core::i32 nz = z + kNeighbor8Z[direction];

            // A particle that wanders too far is reflected rather than lost:
            // letting it escape would waste the whole budget on a walk that can
            // no longer reach the cluster.
            const core::u32 leash = spawnRadius + params.spawnMargin + 2u;
            if (nx < static_cast<core::i32>(border) || nz < static_cast<core::i32>(border) ||
                nx >= static_cast<core::i32>(params.width - border) ||
                nz >= static_cast<core::i32>(params.depth - border) || chebyshev(nx, nz, centerX, centerZ) > leash)
                continue;

            x = nx;
            z = nz;
        }
        if (!landed)
            ++report.abandoned;
    }

    // DLA is connected by construction — a particle only sticks where it
    // touches the cluster — but the thickness carve can round corners into
    // shapes the flood fill sees differently, so the guarantee is re-asserted.
    (void) connectRegions(map, 1u);

    if (outReport != nullptr)
        *outReport = report;
    return map;
}

} // namespace lpl::procgen
