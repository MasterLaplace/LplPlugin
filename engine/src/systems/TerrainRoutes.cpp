#include <lpl/engine/systems/TerrainRoutes.hpp>
#include <lpl/std/vector.hpp>

namespace lpl::engine::systems {

namespace {

/**
 * @brief Floor division of a world unit by a cell size.
 *
 * @warning Floors toward negative infinity rather than toward zero. C division truncates, so with
 * cellSize 4 the cells [-3..-1] and [0..3] would both map to 0 and the one at [-4..-1] would be
 * half as wide as every other -- a body crossing the origin then skips a cell, on one axis, only
 * near the middle of the map.
 *
 * @param value The world unit.
 * @param size  Cell size, never zero.
 * @return The cell ordinal, which may be negative.
 */
[[nodiscard]] core::i32 floorDivide(core::i32 value, core::u32 size) noexcept
{
    const core::i32 divisor = static_cast<core::i32>(size);
    core::i32 quotient = value / divisor;
    if ((value % divisor) != 0 && ((value < 0) != (divisor < 0)))
        --quotient;
    return quotient;
}

} // namespace

void TerrainRoutes::bind(const procgen::Heightfield &field, const history::IPlaceResolver &places,
                         const TerrainRouteParams &params)
{
    _field = &field;
    _places = &places;
    _params = params;
    if (_params.cellSize == 0u)
        _params.cellSize = 1u;
    _roads = procgen::Grid<core::u8>{field.width(), field.depth(), 0u};
    _coarse =
        _params.coarseRatio > 1u ? procgen::reduceHeightfield(field, _params.coarseRatio) : procgen::Heightfield{};
    _paved = 0u;
    _pairs = 0u;
    _planned = 0u;
    _unreachable = 0u;
    _expanded = 0u;
    _coarseExpanded = 0u;
    _corridorCells = 0u;
    _widened = 0u;
}

procgen::RoutedPath TerrainRoutes::plan(core::u32 startX, core::u32 startZ, core::u32 goalX, core::u32 goalZ) const
{
    if (_coarse.empty())
    {
        procgen::RoutedPath flat =
            procgen::routeLeastCost(*_field, &_roads, startX, startZ, goalX, goalZ, _params.cost);
        _expanded += flat.expanded;
        return flat;
    }

    const procgen::HierarchicalRoute cascaded =
        procgen::routeAcrossWorld(_coarse, *_field, _params.coarseRatio, &_roads, startX, startZ, goalX, goalZ,
                                  _params.cost, _params.corridorMargin);
    _expanded += cascaded.fine.expanded;
    _coarseExpanded += cascaded.coarseExpanded;
    _corridorCells += cascaded.corridorCells;
    if (cascaded.widened)
        ++_widened;
    return cascaded.fine;
}

bool TerrainRoutes::toCell(math::Fixed32 x, math::Fixed32 z, core::u32 &outX, core::u32 &outZ) const
{
    if (_field == nullptr || _field->empty())
        return false;

    const core::i32 unitX = static_cast<core::i32>(x.raw() >> 16);
    const core::i32 unitZ = static_cast<core::i32>(z.raw() >> 16);

    const core::i32 cellX = floorDivide(unitX, _params.cellSize) + static_cast<core::i32>(_field->width() / 2u);
    const core::i32 cellZ = floorDivide(unitZ, _params.cellSize) + static_cast<core::i32>(_field->depth() / 2u);

    if (!_field->contains(cellX, cellZ))
        return false;

    outX = static_cast<core::u32>(cellX);
    outZ = static_cast<core::u32>(cellZ);
    return true;
}

void TerrainRoutes::toWorld(core::u32 cellX, core::u32 cellZ, math::Fixed32 &outX, math::Fixed32 &outZ) const
{
    const core::i32 baseX = (static_cast<core::i32>(cellX) - static_cast<core::i32>(_field->width() / 2u)) *
                            static_cast<core::i32>(_params.cellSize);
    const core::i32 baseZ = (static_cast<core::i32>(cellZ) - static_cast<core::i32>(_field->depth() / 2u)) *
                            static_cast<core::i32>(_params.cellSize);

    const core::i32 half = static_cast<core::i32>(_params.cellSize << 15);
    outX = math::Fixed32{(baseX << 16) + half};
    outZ = math::Fixed32{(baseZ << 16) + half};
}

bool TerrainRoutes::cellOf(core::u32 place, core::u32 &outX, core::u32 &outZ) const
{
    if (_places == nullptr)
        return false;

    history::Place resolved;
    if (!_places->resolve(place, resolved))
        return false;
    if (!resolved.located)
        return false;
    return toCell(resolved.x, resolved.z, outX, outZ);
}

core::u32 TerrainRoutes::paveAttested(const core::u32 *places, core::u32 placeCount)
{
    if (_field == nullptr || _places == nullptr || places == nullptr)
        return 0u;

    lpl::pmr::vector<core::u64> laid;

    for (core::u32 i = 0u; i < placeCount; ++i)
    {
        const core::u32 from = places[i];
        core::u32 neighbours[32];
        const core::u32 count = _places->linkedPlaces(from, neighbours, 32u);

        for (core::u32 n = 0u; n < count; ++n)
        {
            const core::u32 to = neighbours[n];
            if (to == from)
                continue;

            const core::u32 low = from < to ? from : to;
            const core::u32 high = from < to ? to : from;
            const core::u64 key = (static_cast<core::u64>(low) << 32) | static_cast<core::u64>(high);

            bool already = false;
            for (core::usize k = 0u; k < laid.size(); ++k)
            {
                if (laid[k] == key)
                {
                    already = true;
                    break;
                }
            }
            if (already)
                continue;
            laid.push_back(key);

            core::u32 startX = 0u;
            core::u32 startZ = 0u;
            core::u32 goalX = 0u;
            core::u32 goalZ = 0u;
            if (!cellOf(low, startX, startZ) || !cellOf(high, goalX, goalZ))
                continue;

            const procgen::RoutedPath path = plan(startX, startZ, goalX, goalZ);
            if (!path.found)
            {
                ++_unreachable;
                continue;
            }

            ++_pairs;

            for (core::usize c = 0u; c < path.cells.size(); ++c)
            {
                core::u8 &cell = _roads[path.cells[c]];
                if (cell == 0u)
                {
                    cell = 1u;
                    ++_paved;
                }
            }
        }
    }

    return _paved;
}

core::u32 TerrainRoutes::route(core::u32 fromPlace, core::u32 toPlace, history::RouteLeg *out, core::u32 capacity) const
{
    if (_field == nullptr || out == nullptr || capacity == 0u)
        return 0u;

    core::u32 startX = 0u;
    core::u32 startZ = 0u;
    core::u32 goalX = 0u;
    core::u32 goalZ = 0u;
    if (!cellOf(fromPlace, startX, startZ) || !cellOf(toPlace, goalX, goalZ))
        return 0u;

    ++_planned;
    const procgen::RoutedPath path = plan(startX, startZ, goalX, goalZ);
    if (!path.found || path.cells.size() < 2u)
    {
        if (!path.found)
            ++_unreachable;
        return 0u;
    }

    lpl::pmr::vector<core::u32> corners;
    const core::u32 width = _field->width();
    core::i32 previousDx = 0;
    core::i32 previousDz = 0;
    for (core::usize i = 1u; i < path.cells.size(); ++i)
    {
        const core::u32 here = path.cells[i];
        const core::u32 before = path.cells[i - 1u];
        const core::i32 dx = static_cast<core::i32>(here % width) - static_cast<core::i32>(before % width);
        const core::i32 dz = static_cast<core::i32>(here / width) - static_cast<core::i32>(before / width);

        const bool turned = (i > 1u) && (dx != previousDx || dz != previousDz);
        const bool last = (i + 1u) == path.cells.size();
        if (turned || last)
            corners.push_back(here);
        previousDx = dx;
        previousDz = dz;
    }

    if (corners.empty())
        return 0u;

    const core::u32 cornerCount = static_cast<core::u32>(corners.size());
    const core::u32 written = cornerCount < capacity ? cornerCount : capacity;
    for (core::u32 i = 0u; i < written; ++i)
    {
        const core::u32 pick =
            cornerCount <= capacity ? i : static_cast<core::u32>(((i + 1u) * cornerCount) / capacity) - 1u;
        const core::u32 cell = corners[pick];
        toWorld(cell % width, cell / width, out[i].x, out[i].z);
    }
    return written;
}

} // namespace lpl::engine::systems
