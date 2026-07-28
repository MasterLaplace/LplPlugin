/**
 * @file Voronoi.cpp
 * @brief Implementation of the jittered-grid Voronoi partition.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Voronoi.hpp>

#include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Where the site of coarse cell (@p cellX, @p cellZ) sits.
 *
 * Derived from the coarse coordinates by hashing, not drawn from a running
 * stream: a site must be computable from its own coordinates alone, so that two
 * chunks generated independently agree about the sites they share. That is the
 * same property that makes the noise seamless.
 */
VoronoiSite siteOf(core::i32 cellX, core::i32 cellZ, const VoronoiParams &params, math::Fixed32 jitter)
{
    const core::u32 hx = ValueNoise2D::hash2(cellX, cellZ, params.seed);
    const core::u32 hz = ValueNoise2D::hash2(cellX, cellZ, params.seed ^ 0x5F356495u);

    // Fractional offsets in [0, 1), scaled by the jitter and centred so a jitter
    // of 0 puts the site exactly at the coarse cell's middle.
    const math::Fixed32 offsetX =
        (math::Fixed32::fromRaw(static_cast<core::i32>(hx & 0xFFFFu)) - math::Fixed32::half()) * jitter;
    const math::Fixed32 offsetZ =
        (math::Fixed32::fromRaw(static_cast<core::i32>(hz & 0xFFFFu)) - math::Fixed32::half()) * jitter;

    const math::Fixed32 size = math::Fixed32::fromInt(static_cast<core::i32>(params.cellSize));
    VoronoiSite site;
    site.x = (math::Fixed32::fromInt(cellX) + math::Fixed32::half() + offsetX) * size;
    site.z = (math::Fixed32::fromInt(cellZ) + math::Fixed32::half() + offsetZ) * size;
    return site;
}

/**
 * @brief Distance from a point to a site under the chosen metric.
 *
 * Euclidean is returned SQUARED and the other two are not, which is fine because
 * the value is only ever compared against other distances computed the same way —
 * and squaring is monotone, so it cannot change which site is nearest. Taking a
 * root to make the three commensurable would cost precision and buy nothing.
 */
math::Fixed32 metricDistance(math::Fixed32 px, math::Fixed32 pz, const VoronoiSite &site, DistanceMetric metric)
{
    const math::Fixed32 dx = (px - site.x).abs();
    const math::Fixed32 dz = (pz - site.z).abs();
    switch (metric)
    {
    case DistanceMetric::Manhattan: return dx + dz;
    case DistanceMetric::Chebyshev: return dx > dz ? dx : dz;
    case DistanceMetric::Euclidean: break;
    }
    return dx * dx + dz * dz;
}

} // namespace

VoronoiDiagram computeVoronoi(const VoronoiParams &params)
{
    VoronoiDiagram diagram;
    if (params.width == 0u || params.depth == 0u || params.cellSize == 0u)
        return diagram;

    const math::Fixed32 jitter = math::Fixed32::fromFloat(params.jitter);
    const core::i32 coarseWidth = static_cast<core::i32>((params.width + params.cellSize - 1u) / params.cellSize);
    const core::i32 coarseDepth = static_cast<core::i32>((params.depth + params.cellSize - 1u) / params.cellSize);

    // One region per coarse cell, indexed in scan order so a region id is
    // reproducible from its coarse coordinates alone.
    //
    // A region id is a u16 and 0xFFFF means "no region", so a partition finer than
    // that would wrap ids into the sentinel and silently mark cells unclaimed.
    // Refusing outright beats producing a diagram whose region 65535 is invisible.
    const core::u32 requested = static_cast<core::u32>(coarseWidth) * static_cast<core::u32>(coarseDepth);
    if (requested >= static_cast<core::u32>(kNoRegion))
        return diagram;

    diagram.regionCount = requested;
    diagram.sites.reserve(diagram.regionCount);
    for (core::i32 cz = 0; cz < coarseDepth; ++cz)
    {
        for (core::i32 cx = 0; cx < coarseWidth; ++cx)
        {
            VoronoiSite site = siteOf(cx, cz, params, jitter);
            site.region = static_cast<core::u16>(cz * coarseWidth + cx);
            diagram.sites.push_back(site);
        }
    }

    diagram.regions = RegionMap{params.width, params.depth, kNoRegion};

    const math::Fixed32 warpStrength = math::Fixed32::fromFloat(params.warpStrength);
    // The warp displaces the query point, so a warped point can land outside its
    // own coarse cell and the 3x3 block stops being a sufficient search. Widening
    // the block by however many coarse cells the displacement can cross restores
    // the guarantee.
    const core::i32 warpReach =
        warpStrength.raw() <= 0 ?
            0 :
            static_cast<core::i32>(warpStrength.toInt()) / static_cast<core::i32>(params.cellSize) + 1;
    const core::i32 reach = 1 + warpReach;

    for (core::u32 z = 0u; z < params.depth; ++z)
    {
        for (core::u32 x = 0u; x < params.width; ++x)
        {
            math::Fixed32 px = math::Fixed32::fromInt(static_cast<core::i32>(x));
            math::Fixed32 pz = math::Fixed32::fromInt(static_cast<core::i32>(z));
            ValueNoise2D::warp(px, pz, params.seed ^ 0x1B873593u, warpStrength,
                               math::Fixed32::one() / math::Fixed32::fromInt(static_cast<core::i32>(params.cellSize)));

            const core::i32 homeX = px.toInt() / static_cast<core::i32>(params.cellSize);
            const core::i32 homeZ = pz.toInt() / static_cast<core::i32>(params.cellSize);

            math::Fixed32 best = math::Fixed32::max();
            core::u16 bestRegion = kNoRegion;

            // The block around the home coarse cell is enough: a jittered site
            // never leaves its own cell, so no site outside the block can be
            // closer than the nearest one inside it.
            for (core::i32 dz = -reach; dz <= reach; ++dz)
            {
                for (core::i32 dx = -reach; dx <= reach; ++dx)
                {
                    const core::i32 cx = homeX + dx;
                    const core::i32 cz = homeZ + dz;
                    if (cx < 0 || cz < 0 || cx >= coarseWidth || cz >= coarseDepth)
                        continue;
                    const core::u32 region = static_cast<core::u32>(cz * coarseWidth + cx);
                    const math::Fixed32 distance = metricDistance(px, pz, diagram.sites[region], params.metric);
                    // Strictly less: ties go to the first in scan order, so the
                    // partition is the same on every target.
                    if (distance < best)
                    {
                        best = distance;
                        bestRegion = static_cast<core::u16>(region);
                    }
                }
            }
            diagram.regions.at(x, z) = bestRegion;
        }
    }
    return diagram;
}

Grid<core::u8> regionBorders(const VoronoiDiagram &diagram)
{
    Grid<core::u8> borders{diagram.regions.width(), diagram.regions.depth(), 0u};

    for (core::u32 z = 0u; z < diagram.regions.depth(); ++z)
    {
        for (core::u32 x = 0u; x < diagram.regions.width(); ++x)
        {
            const core::u16 here = diagram.regions.at(x, z);
            for (core::u32 n = 0u; n < 4u; ++n)
            {
                const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                if (!diagram.regions.contains(nx, nz))
                    continue;
                if (diagram.regions.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) != here)
                {
                    borders.at(x, z) = 1u;
                    break;
                }
            }
        }
    }
    return borders;
}

Grid<math::Fixed32> regionDistanceField(const VoronoiDiagram &diagram, DistanceMetric metric)
{
    Grid<math::Fixed32> field{diagram.regions.width(), diagram.regions.depth(), math::Fixed32::zero()};

    for (core::u32 z = 0u; z < diagram.regions.depth(); ++z)
    {
        for (core::u32 x = 0u; x < diagram.regions.width(); ++x)
        {
            const core::u16 region = diagram.regions.at(x, z);
            if (region == kNoRegion || region >= diagram.sites.size())
                continue;
            field.at(x, z) =
                metricDistance(math::Fixed32::fromInt(static_cast<core::i32>(x)),
                               math::Fixed32::fromInt(static_cast<core::i32>(z)), diagram.sites[region], metric);
        }
    }
    return field;
}

void countRegionCells(const VoronoiDiagram &diagram, core::u32 *outCounts)
{
    if (outCounts == nullptr)
        return;
    for (core::u32 i = 0u; i < diagram.regionCount; ++i)
        outCounts[i] = 0u;
    for (core::u32 i = 0u; i < diagram.regions.cellCount(); ++i)
    {
        const core::u16 region = diagram.regions[i];
        if (region < diagram.regionCount)
            ++outCounts[region];
    }
}

core::u32 foldRegionMap(const RegionMap &map)
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        hash = (hash ^ static_cast<core::u32>(map[i])) * kFnv1aPrime;
    return hash;
}

} // namespace lpl::procgen
