/**
 * @file Settlement.cpp
 * @brief Implementation of district, road and plot generation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Settlement.hpp>

#include <lpl/procgen/Random.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

constexpr bool isWalkableCell(SettlementCell cell) noexcept
{
    return cell == SettlementCell::Road || cell == SettlementCell::Plaza;
}

/// Paints a road cell unless the ground refuses it.
void placeRoad(SettlementMap &map, core::i32 x, core::i32 z, core::u32 thickness, core::u32 &outCells)
{
    const core::i32 r = static_cast<core::i32>(thickness);
    for (core::i32 dz = -r; dz <= r; ++dz)
    {
        for (core::i32 dx = -r; dx <= r; ++dx)
        {
            const core::i32 cx = x + dx;
            const core::i32 cz = z + dz;
            if (!map.contains(cx, cz))
                continue;
            SettlementCell &cell = map.at(static_cast<core::u32>(cx), static_cast<core::u32>(cz));
            // Blocked ground stays blocked: a road must route around a cliff,
            // not pretend it is not there.
            if (cell == SettlementCell::Blocked || cell == SettlementCell::Road)
                continue;
            cell = SettlementCell::Road;
            ++outCells;
        }
    }
}

/// Draws an L-shaped street between two points.
void connectWithRoad(SettlementMap &map, core::u32 fromX, core::u32 fromZ, core::u32 toX, core::u32 toZ,
                     core::u32 thickness, core::u32 &outCells)
{
    const core::u32 x0 = fromX < toX ? fromX : toX;
    const core::u32 x1 = fromX < toX ? toX : fromX;
    for (core::u32 x = x0; x <= x1; ++x)
        placeRoad(map, static_cast<core::i32>(x), static_cast<core::i32>(fromZ), thickness, outCells);

    const core::u32 z0 = fromZ < toZ ? fromZ : toZ;
    const core::u32 z1 = fromZ < toZ ? toZ : fromZ;
    for (core::u32 z = z0; z <= z1; ++z)
        placeRoad(map, static_cast<core::i32>(toX), static_cast<core::i32>(z), thickness, outCells);
}

/// Does any 4-neighbour carry a street?
bool facesRoad(const SettlementMap &map, core::u32 x, core::u32 z)
{
    for (core::u32 n = 0u; n < 4u; ++n)
    {
        const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
        const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
        if (map.contains(nx, nz) && isWalkableCell(map.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz))))
            return true;
    }
    return false;
}

/// Can a footprint of this size be placed here?
bool plotFits(const SettlementMap &map, core::u32 x, core::u32 z, core::u32 width, core::u32 depth)
{
    for (core::u32 cz = z; cz < z + depth; ++cz)
        for (core::u32 cx = x; cx < x + width; ++cx)
        {
            if (!map.contains(static_cast<core::i32>(cx), static_cast<core::i32>(cz)))
                return false;
            if (map.at(cx, cz) != SettlementCell::Empty)
                return false;
        }
    return true;
}

/**
 * @brief Joins every disconnected piece of the street network into one.
 *
 * Labels the walkable components, then chains their representatives with
 * L-corridors. Chaining rather than building a minimum spanning tree digs
 * slightly longer streets, but it is linear in the number of components and
 * cannot leave one behind.
 */
void linkWalkableComponents(SettlementMap &map, core::u32 thickness, core::u32 &outRoadCells)
{
    Grid<core::u32> labels{map.width(), map.depth(), 0xFFFFFFFFu};
    lpl::pmr::vector<core::u32> representatives;
    lpl::pmr::vector<core::u32> queue;

    core::u32 componentCount = 0u;
    for (core::u32 start = 0u; start < map.cellCount(); ++start)
    {
        if (!isWalkableCell(map[start]) || labels[start] != 0xFFFFFFFFu)
            continue;

        representatives.push_back(start);
        queue.clear();
        queue.push_back(start);
        labels[start] = componentCount;

        for (core::u32 head = 0u; head < queue.size(); ++head)
        {
            const core::u32 cell = queue[head];
            const core::u32 x = cell % map.width();
            const core::u32 z = cell / map.width();
            for (core::u32 n = 0u; n < 4u; ++n)
            {
                const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
                const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
                if (!map.contains(nx, nz))
                    continue;
                const core::u32 index = map.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                if (labels[index] != 0xFFFFFFFFu || !isWalkableCell(map[index]))
                    continue;
                labels[index] = componentCount;
                queue.push_back(index);
            }
        }
        ++componentCount;
    }

    for (core::u32 i = 1u; i < representatives.size(); ++i)
    {
        const core::u32 from = representatives[i - 1u];
        const core::u32 to = representatives[i];
        connectWithRoad(map, from % map.width(), from / map.width(), to % map.width(), to / map.width(), thickness,
                        outRoadCells);
    }
}

/// The core layout, shared by the flat and terrain-aware entry points.
SettlementMap layout(const SettlementParams &params, SettlementMap map, lpl::pmr::vector<BuildingPlot> *outPlots,
                     SettlementReport *outReport)
{
    SettlementReport report;
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        if (map[i] == SettlementCell::Blocked)
            ++report.blockedCells;

    // ── Districts ───────────────────────────────────────────────────────────
    VoronoiParams voronoi;
    voronoi.width = params.width;
    voronoi.depth = params.depth;
    voronoi.seed = deriveStream(params.seed, 0xD157u).state();
    voronoi.cellSize = params.districtSize;
    voronoi.jitter = 0.7f;
    const VoronoiDiagram districts = computeVoronoi(voronoi);
    report.districts = districts.regionCount;

    if (districts.regionCount == 0u)
    {
        if (outReport != nullptr)
            *outReport = report;
        return map;
    }

    // ── Roads along the district borders ────────────────────────────────────
    const Grid<core::u8> borders = regionBorders(districts);
    for (core::u32 z = 0u; z < map.depth(); ++z)
        for (core::u32 x = 0u; x < map.width(); ++x)
            if (borders.at(x, z) != 0u)
                placeRoad(map, static_cast<core::i32>(x), static_cast<core::i32>(z), params.roadWidth / 2u,
                          report.roadCells);

    // ── Plazas, and a spur linking each centre to the border network ────────
    for (core::u32 region = 0u; region < districts.regionCount; ++region)
    {
        const VoronoiSite &site = districts.sites[region];
        const core::i32 centerX = site.x.toInt();
        const core::i32 centerZ = site.z.toInt();
        if (!map.contains(centerX, centerZ))
            continue;

        // Plaza: open ground at the heart of the district.
        const core::i32 r = static_cast<core::i32>(params.plazaRadius);
        for (core::i32 dz = -r; dz <= r; ++dz)
        {
            for (core::i32 dx = -r; dx <= r; ++dx)
            {
                const core::i32 cx = centerX + dx;
                const core::i32 cz = centerZ + dz;
                if (!map.contains(cx, cz))
                    continue;
                SettlementCell &cell = map.at(static_cast<core::u32>(cx), static_cast<core::u32>(cz));
                if (cell == SettlementCell::Blocked || cell == SettlementCell::Road)
                    continue;
                cell = SettlementCell::Plaza;
                ++report.plazaCells;
            }
        }

        // Spur: walk outward until the border network is met, then join. A
        // district whose centre is enclaved would otherwise have buildings
        // nobody can reach.
        core::i32 probeX = centerX;
        core::i32 probeZ = centerZ;
        core::u32 guard = 0u;
        while (map.contains(probeX, probeZ) && guard < params.districtSize * 2u)
        {
            if (map.at(static_cast<core::u32>(probeX), static_cast<core::u32>(probeZ)) == SettlementCell::Road)
                break;
            ++probeX;
            ++guard;
        }
        if (map.contains(probeX, probeZ))
            connectWithRoad(map, static_cast<core::u32>(centerX), static_cast<core::u32>(centerZ),
                            static_cast<core::u32>(probeX), static_cast<core::u32>(probeZ), params.roadWidth / 2u,
                            report.roadCells);
    }

    // ── Plots facing a street ───────────────────────────────────────────────
    Random random = deriveStream(params.seed, 0xB1D5u);
    const math::Fixed32 density = math::Fixed32::fromFloat(params.plotDensity);
    const core::u32 minPlot = params.minPlot == 0u ? 1u : params.minPlot;
    const core::u32 maxPlot = params.maxPlot < minPlot ? minPlot : params.maxPlot;

    for (core::u32 z = 0u; z < map.depth(); ++z)
    {
        for (core::u32 x = 0u; x < map.width(); ++x)
        {
            if (map.at(x, z) != SettlementCell::Empty)
                continue;
            // Facing a street is what makes a plot a plot; without the rule
            // buildings end up stranded inside blocks.
            if (!facesRoad(map, x, z))
                continue;
            if (!random.chance(density))
                continue;

            const core::u32 width =
                static_cast<core::u32>(random.range(static_cast<core::i32>(minPlot), static_cast<core::i32>(maxPlot)));
            const core::u32 depth =
                static_cast<core::u32>(random.range(static_cast<core::i32>(minPlot), static_cast<core::i32>(maxPlot)));
            if (!plotFits(map, x, z, width, depth))
                continue;

            for (core::u32 cz = z; cz < z + depth; ++cz)
                for (core::u32 cx = x; cx < x + width; ++cx)
                    map.at(cx, cz) = SettlementCell::Plot;

            if (outPlots != nullptr)
                outPlots->push_back(BuildingPlot{x, z, width, depth, districts.regions.at(x, z)});
            ++report.plots;
        }
    }

    // Roads traced along Voronoi borders can meet only diagonally, and a
    // district whose spur failed to reach the network is enclaved. Neither is
    // acceptable: this module's rule is that a bottom-up layout guarantees
    // nothing until a repair pass has said otherwise.
    linkWalkableComponents(map, params.roadWidth / 2u, report.roadCells);

    report.roadsConnected = areRoadsConnected(map);
    if (outReport != nullptr)
        *outReport = report;
    return map;
}

} // namespace

SettlementMap generateSettlement(const SettlementParams &params, lpl::pmr::vector<BuildingPlot> *outPlots,
                                 SettlementReport *outReport)
{
    if (params.width == 0u || params.depth == 0u)
        return SettlementMap{};
    return layout(params, SettlementMap{params.width, params.depth, SettlementCell::Empty}, outPlots, outReport);
}

SettlementMap generateSettlementOnTerrain(const SettlementParams &params, const Heightfield &terrain,
                                          lpl::pmr::vector<BuildingPlot> *outPlots, SettlementReport *outReport)
{
    if (params.width == 0u || params.depth == 0u)
        return SettlementMap{};
    if (terrain.width() != params.width || terrain.depth() != params.depth)
        return SettlementMap{};

    SettlementMap map{params.width, params.depth, SettlementCell::Empty};

    // Mark the unbuildable ground BEFORE anything is placed, so roads route
    // around it instead of being erased afterwards.
    const math::Fixed32 maxSlope = math::Fixed32::fromFloat(params.maxSlope);
    const math::Fixed32 minHeight = math::Fixed32::fromFloat(params.minHeight);
    for (core::u32 z = 0u; z < map.depth(); ++z)
        for (core::u32 x = 0u; x < map.width(); ++x)
            if (terrain.at(x, z) < minHeight || slopeAt(terrain, x, z) > maxSlope)
                map.at(x, z) = SettlementCell::Blocked;

    return layout(params, map, outPlots, outReport);
}

bool areRoadsConnected(const SettlementMap &map)
{
    lpl::pmr::vector<core::u32> queue;
    Grid<core::u8> visited{map.width(), map.depth(), 0u};

    core::u32 total = 0u;
    core::i32 start = -1;
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        if (isWalkableCell(map[i]))
        {
            ++total;
            if (start < 0)
                start = static_cast<core::i32>(i);
        }

    if (total == 0u)
        return true; // nothing to disconnect

    queue.push_back(static_cast<core::u32>(start));
    visited[static_cast<core::u32>(start)] = 1u;
    core::u32 reached = 0u;

    for (core::u32 head = 0u; head < queue.size(); ++head)
    {
        const core::u32 cell = queue[head];
        ++reached;
        const core::u32 x = cell % map.width();
        const core::u32 z = cell / map.width();
        for (core::u32 n = 0u; n < 4u; ++n)
        {
            const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[n];
            const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[n];
            if (!map.contains(nx, nz))
                continue;
            const core::u32 index = map.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
            if (visited[index] != 0u || !isWalkableCell(map[index]))
                continue;
            visited[index] = 1u;
            queue.push_back(index);
        }
    }
    return reached == total;
}

core::u32 foldSettlement(const SettlementMap &map)
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < map.cellCount(); ++i)
        hash = (hash ^ static_cast<core::u32>(map[i])) * kFnv1aPrime;
    return hash;
}

} // namespace lpl::procgen
