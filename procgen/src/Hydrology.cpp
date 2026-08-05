/**
 * @file Hydrology.cpp
 * @brief Implementation of depression filling, flow routing, rivers and moisture.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Hydrology.hpp>

#include <lpl/math/FixedMath.hpp>
#include <lpl/procgen/ValueNoise.hpp>
#include <lpl/std/vector.hpp>

namespace lpl::procgen {

namespace {

/// One Q16.16 tick: the smallest rise that still makes a slope.
constexpr core::i32 kFillEpsilon = 1;

/// A cell waiting its turn in the flood, keyed by the height it was reached at.
struct FloodEntry {
    core::i32 height;
    core::u32 cell;
};

/**
 * @brief Total order on flood entries: height first, then cell index.
 *
 * The index tiebreak is not cosmetic. Priority-Flood pops cells at equal height
 * constantly — a plateau is nothing but equal heights — and if their order were
 * left to the heap's internal layout it would depend on insertion history, so
 * two targets could resolve the same plateau differently. Ordering by index
 * makes the pop sequence a function of the terrain alone.
 */
constexpr bool precedes(const FloodEntry &a, const FloodEntry &b) noexcept
{
    return a.height != b.height ? a.height < b.height : a.cell < b.cell;
}

/**
 * @class FloodHeap
 * @brief A binary min-heap over @ref FloodEntry.
 *
 * Hand-rolled because `std::priority_queue` is not available freestanding, and
 * because the comparison has to be exactly @ref precedes for the flood to stay
 * reproducible.
 */
class FloodHeap {
public:
    void reserve(core::usize capacity) { _items.reserve(capacity); }
    [[nodiscard]] bool empty() const { return _items.empty(); }

    void push(FloodEntry entry)
    {
        _items.push_back(entry);
        core::u32 child = static_cast<core::u32>(_items.size()) - 1u;
        while (child != 0u)
        {
            const core::u32 parent = (child - 1u) / 2u;
            if (!precedes(_items[child], _items[parent]))
                break;
            const FloodEntry swap = _items[parent];
            _items[parent] = _items[child];
            _items[child] = swap;
            child = parent;
        }
    }

    [[nodiscard]] FloodEntry pop()
    {
        const FloodEntry top = _items[0];
        _items[0] = _items[_items.size() - 1u];
        _items.pop_back();

        const core::u32 count = static_cast<core::u32>(_items.size());
        core::u32 node = 0u;
        for (;;)
        {
            const core::u32 left = node * 2u + 1u;
            const core::u32 right = left + 1u;
            core::u32 smallest = node;
            if (left < count && precedes(_items[left], _items[smallest]))
                smallest = left;
            if (right < count && precedes(_items[right], _items[smallest]))
                smallest = right;
            if (smallest == node)
                break;
            const FloodEntry swap = _items[smallest];
            _items[smallest] = _items[node];
            _items[node] = swap;
            node = smallest;
        }
        return top;
    }

private:
    lpl::pmr::vector<FloodEntry> _items;
};

/**
 * @brief Priority-Flood with an epsilon rise.
 *
 * @param field    Surface to fill in place.
 * @param outOrder Receives the cells in the order they were resolved, which is
 *                 ascending filled height. May be null.
 * @return Number of cells raised.
 */
core::u32 priorityFlood(Heightfield &field, lpl::pmr::vector<core::u32> *outOrder)
{
    if (field.empty())
        return 0u;

    const core::u32 width = field.width();
    const core::u32 depth = field.depth();
    const core::u32 count = field.cellCount();

    Grid<core::u8> resolved{width, depth, 0u};
    FloodHeap heap;
    heap.reserve(width * 2u + depth * 2u);
    if (outOrder != nullptr)
    {
        outOrder->clear();
        outOrder->reserve(count);
    }

    // The border is where water leaves the map, so it is where the flood starts.
    // Its cells keep their own height: nothing outside can hold them up.
    for (core::u32 x = 0u; x < width; ++x)
    {
        const core::u32 top = field.index(x, 0u);
        const core::u32 bottom = field.index(x, depth - 1u);
        resolved[top] = 1u;
        heap.push(FloodEntry{field[top].raw(), top});
        if (bottom != top)
        {
            resolved[bottom] = 1u;
            heap.push(FloodEntry{field[bottom].raw(), bottom});
        }
    }
    for (core::u32 z = 0u; z < depth; ++z)
    {
        const core::u32 left = field.index(0u, z);
        const core::u32 right = field.index(width - 1u, z);
        if (resolved[left] == 0u)
        {
            resolved[left] = 1u;
            heap.push(FloodEntry{field[left].raw(), left});
        }
        if (resolved[right] == 0u)
        {
            resolved[right] = 1u;
            heap.push(FloodEntry{field[right].raw(), right});
        }
    }

    core::u32 raised = 0u;
    while (!heap.empty())
    {
        const FloodEntry current = heap.pop();
        if (outOrder != nullptr)
            outOrder->push_back(current.cell);

        const core::u32 x = current.cell % width;
        const core::u32 z = current.cell / width;

        for (core::u32 n = 0u; n < 8u; ++n)
        {
            const core::i32 nx = static_cast<core::i32>(x) + kNeighbor8X[n];
            const core::i32 nz = static_cast<core::i32>(z) + kNeighbor8Z[n];
            if (!field.contains(nx, nz))
                continue;
            const core::u32 index = field.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
            if (resolved[index] != 0u)
                continue;

            resolved[index] = 1u;
            // Anything at or below where we came from is inside a depression:
            // lift it one tick above, which both removes the sink and leaves a
            // definite direction for water to follow out of the basin.
            const core::i32 spill = current.height + kFillEpsilon;
            if (field[index].raw() <= spill)
            {
                field[index] = math::Fixed32::fromRaw(spill);
                ++raised;
            }
            heap.push(FloodEntry{field[index].raw(), index});
        }
    }
    return raised;
}

} // namespace

core::u32 fillDepressions(Heightfield &field) { return priorityFlood(field, nullptr); }

DrainageNetwork computeDrainage(const Heightfield &field)
{
    DrainageNetwork network;
    if (field.empty())
        return network;

    // Route over a filled copy: the caller's terrain keeps its basins (they are
    // lakes, and a lake is a feature) while routing sees a surface where every
    // cell has somewhere to send its water.
    network.filled = field;
    lpl::pmr::vector<core::u32> order;
    network.raisedCells = priorityFlood(network.filled, &order);

    network.direction = FlowDirection{field.width(), field.depth(), kNoFlow};
    network.accumulation = FlowAccumulation{field.width(), field.depth(), 1u};

    // ── Steepest-descent direction per cell, over eight neighbours ───────────
    for (core::u32 z = 0u; z < field.depth(); ++z)
    {
        for (core::u32 x = 0u; x < field.width(); ++x)
        {
            const math::Fixed32 here = network.filled.at(x, z);
            math::Fixed32 steepest = math::Fixed32::zero();
            core::u8 chosen = kNoFlow;

            for (core::u8 n = 0u; n < 8u; ++n)
            {
                const core::i32 nx = static_cast<core::i32>(x) + kNeighbor8X[n];
                const core::i32 nz = static_cast<core::i32>(z) + kNeighbor8Z[n];
                if (!field.contains(nx, nz))
                    continue;
                const math::Fixed32 drop =
                    here - network.filled.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                if (drop.raw() <= 0)
                    continue;

                // A diagonal step falls the same distance over a longer run, so
                // comparing raw drops would bias every channel onto diagonals.
                const math::Fixed32 slope = n < 4u ? drop : drop * math::kInvSqrt2;
                // Strictly greater: the first of several equally steep
                // neighbours wins, and kNeighbor8 order is fixed, so the choice
                // is the same everywhere.
                if (slope > steepest)
                {
                    steepest = slope;
                    chosen = n;
                }
            }
            network.direction.at(x, z) = chosen;
        }
    }

    // ── Accumulate, highest cell first ──────────────────────────────────────
    //
    // The flood already visited every cell in ascending filled height, and after
    // filling a cell always drains to one strictly lower, so walking that order
    // backwards is a topological order of the flow graph: a cell's own inflow is
    // final before it passes water on. That is exactly what an explicit sort used
    // to be for, at O(n) instead of O(n^2).
    for (core::u32 i = order.size(); i-- > 0u;)
    {
        const core::u32 cell = order[i];
        const core::u8 direction = network.direction[cell];
        if (direction == kNoFlow)
            continue;

        const core::u32 x = cell % field.width();
        const core::u32 z = cell / field.width();
        const core::u32 nx = static_cast<core::u32>(static_cast<core::i32>(x) + kNeighbor8X[direction]);
        const core::u32 nz = static_cast<core::u32>(static_cast<core::i32>(z) + kNeighbor8Z[direction]);
        network.accumulation.at(nx, nz) += network.accumulation[cell];
    }

    for (core::u32 i = 0u; i < network.accumulation.cellCount(); ++i)
        if (network.accumulation[i] > network.maxAccumulation)
            network.maxAccumulation = network.accumulation[i];

    return network;
}

Grid<core::u8> lakeMask(const DrainageNetwork &network, const Heightfield &original, core::f32 minDepth)
{
    if (network.filled.width() != original.width() || network.filled.depth() != original.depth())
        return Grid<core::u8>{};

    // Deeper than minDepth, not merely raised. Priority-Flood lifts every cell of
    // a basin by at least one tick, so "raised" includes the whole endorheic
    // catchment down to films of water a Q16.16 tick deep — 20% of a measured
    // 64x64 world. A lake is where water STANDS.
    const math::Fixed32 floor = math::Fixed32::fromFloat(minDepth);
    Grid<core::u8> mask{original.width(), original.depth(), 0u};
    for (core::u32 i = 0u; i < mask.cellCount(); ++i)
        mask[i] = (network.filled[i] - original[i]) >= floor ? 1u : 0u;
    return mask;
}

Heightfield lakeDepth(const DrainageNetwork &network, const Heightfield &original)
{
    if (network.filled.width() != original.width() || network.filled.depth() != original.depth())
        return Heightfield{};

    Heightfield depth{original.width(), original.depth(), math::Fixed32::zero()};
    for (core::u32 i = 0u; i < depth.cellCount(); ++i)
    {
        const math::Fixed32 delta = network.filled[i] - original[i];
        depth[i] = delta.raw() > 0 ? delta : math::Fixed32::zero();
    }
    return depth;
}

Grid<core::u8> riverMask(const DrainageNetwork &network, core::f32 density)
{
    Grid<core::u8> mask{network.accumulation.width(), network.accumulation.depth(), 0u};
    if (network.maxAccumulation == 0u || mask.cellCount() == 0u)
        return mask;

    // The threshold is a QUANTILE of the accumulation distribution, not a share
    // of the largest flow. That distinction is the module's recurring bug in yet
    // another costume: accumulation is heavy-tailed and its spread depends on how
    // big the map is, so a fixed share of the maximum picks a wildly different
    // share of the map at different sizes. Measured on the same recipe, a
    // threshold of 2% of the maximum made 3.7% of a 128x128 world river and 34%
    // of a 24x24 one — the small world read as a flooded plain. A quantile is
    // scale-free by construction: ask for 4% of the cells and get 4%, at any size.
    //
    // Selected by counting sort rather than by sorting: accumulation is an
    // integer bounded by the cell count, so a histogram walked from the top is
    // O(cells) and has no comparison order to get wrong.
    const core::u32 cells = mask.cellCount();
    core::u32 target = static_cast<core::u32>(
        (static_cast<core::i64>(math::Fixed32::fromFloat(density).raw()) * static_cast<core::i64>(cells)) >> 16);
    if (target == 0u)
        return mask; // a density of zero is a world without visible water

    lpl::pmr::vector<core::u32> histogram(static_cast<core::usize>(network.maxAccumulation) + 1u, 0u);
    for (core::u32 i = 0u; i < cells; ++i)
        ++histogram[network.accumulation[i]];

    // Walk down from the strongest flow until the target share is covered. Ties
    // are kept whole: cutting inside a group of equal accumulations would make
    // the result depend on visit order, which is exactly what must never happen.
    core::u32 threshold = network.maxAccumulation;
    core::u32 covered = 0u;
    for (core::u32 value = network.maxAccumulation; value > 0u; --value)
    {
        covered += histogram[value];
        threshold = value;
        if (covered >= target)
            break;
    }

    // A river always means water actually converged, never merely existing: a
    // flat or tiny map would otherwise call its every cell a river.
    if (threshold < 2u)
        threshold = 2u;

    for (core::u32 i = 0u; i < cells; ++i)
        mask[i] = network.accumulation[i] >= threshold ? 1u : 0u;
    return mask;
}

core::u32 carveRivers(Heightfield &field, const DrainageNetwork &network, const RiverParams &params)
{
    if (field.empty() || network.maxAccumulation == 0u)
        return 0u;

    const Grid<core::u8> mask = riverMask(network, params.density);
    const math::Fixed32 maxDepth = math::Fixed32::fromFloat(params.carveDepth);

    // ── Stream power per river cell: sqrt(area) * channel slope ─────────────
    //
    // The slope is the fall to the cell's own downstream neighbour, so it is the
    // slope of the channel rather than of the surrounding land. Held in 64-bit
    // integers: sqrt(area) reaches a few thousand and a raw slope a few million,
    // whose product leaves Fixed32's range long before it leaves i64's.
    lpl::pmr::vector<core::i64> power(field.cellCount(), core::i64{0});
    core::i64 maxPower = 0;

    for (core::u32 i = 0u; i < field.cellCount(); ++i)
    {
        if (mask[i] == 0u)
            continue;
        const core::u8 direction = network.direction[i];
        if (direction == kNoFlow)
            continue;

        const core::u32 x = i % field.width();
        const core::u32 z = i / field.width();
        const core::i32 nx = static_cast<core::i32>(x) + kNeighbor8X[direction];
        const core::i32 nz = static_cast<core::i32>(z) + kNeighbor8Z[direction];
        if (!field.contains(nx, nz))
            continue;

        math::Fixed32 drop =
            network.filled[i] - network.filled.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
        if (drop.raw() <= 0)
            continue;
        if (direction >= 4u)
            drop = drop * math::kInvSqrt2;

        const core::i64 value =
            static_cast<core::i64>(math::integerSqrt(network.accumulation[i])) * static_cast<core::i64>(drop.raw());
        power[i] = value;
        if (value > maxPower)
            maxPower = value;
    }

    if (maxPower == 0)
        return 0u; // the network exists but nothing in it has any fall to exploit

    core::u32 carved = 0u;
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
    {
        if (power[i] == 0)
            continue;
        // Normalise against the strongest cell, so carveDepth keeps meaning
        // "how deep the strongest river cuts".
        const math::Fixed32 share = math::Fixed32::fromRaw(static_cast<core::i32>((power[i] << 16) / maxPower));
        field[i] = field[i] - maxDepth * share;
        ++carved;
    }

    if (params.smoothing != 0u)
        smoothHeights(field, params.smoothing);

    return carved;
}

Grid<core::u32> distanceToSea(const Heightfield &field, math::Fixed32 seaLevel)
{
    Grid<core::u8> sea{field.width(), field.depth(), core::u8{0}};
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
        sea[i] = field[i] <= seaLevel ? 1u : 0u;
    return chamferDistance(sea);
}

Grid<core::u32> chamferDistance(const Grid<core::u8> &seeds)
{
    Grid<core::u32> seaDistance{seeds.width(), seeds.depth(), kUnreachedFromSea};
    if (seeds.empty())
        return seaDistance;

    for (core::u32 i = 0u; i < seeds.cellCount(); ++i)
        if (seeds[i] != 0u)
            seaDistance[i] = 0u;

    const auto relax = [&seaDistance](core::u32 x, core::u32 z, core::i32 dx, core::i32 dz) {
        const core::i32 nx = static_cast<core::i32>(x) + dx;
        const core::i32 nz = static_cast<core::i32>(z) + dz;
        if (!seaDistance.contains(nx, nz))
            return;
        const core::u32 candidate = seaDistance.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) + 1u;
        if (candidate < seaDistance.at(x, z))
            seaDistance.at(x, z) = candidate;
    };

    // Forward sweep resolves distances coming from the west and north, backward
    // sweep the other two. Two passes suffice for a 4-connected chamfer.
    for (core::u32 z = 0u; z < seeds.depth(); ++z)
        for (core::u32 x = 0u; x < seeds.width(); ++x)
        {
            relax(x, z, -1, 0);
            relax(x, z, 0, -1);
        }
    for (core::u32 z = seeds.depth(); z-- > 0u;)
        for (core::u32 x = seeds.width(); x-- > 0u;)
        {
            relax(x, z, 1, 0);
            relax(x, z, 0, 1);
        }
    return seaDistance;
}

Heightfield computeMoisture(const Heightfield &field, const DrainageNetwork &network, const MoistureParams &params)
{
    Heightfield moisture{field.width(), field.depth(), math::Fixed32::zero()};
    if (field.empty())
        return moisture;

    math::Fixed32 lowest{};
    math::Fixed32 highest{};
    (void) heightRange(field, lowest, highest);
    const math::Fixed32 span = highest - lowest;

    const math::Fixed32 rainfallWeight = math::Fixed32::fromFloat(params.rainfallWeight);
    const math::Fixed32 flowWeight = math::Fixed32::fromFloat(params.flowWeight);
    const math::Fixed32 altitudeWeight = math::Fixed32::fromFloat(params.altitudeWeight);
    const math::Fixed32 coastWeight = math::Fixed32::fromFloat(params.coastWeight);
    const math::Fixed32 seaLevel = math::Fixed32::fromFloat(params.seaLevel);

    // Map-relative scales. The longer axis is the reference, so a rectangular map
    // gets belts of the same real width along both axes rather than stretched ones.
    const core::u32 longAxis = field.width() > field.depth() ? field.width() : field.depth();
    const math::Fixed32 axis = math::Fixed32::fromInt(static_cast<core::i32>(longAxis));
    const math::Fixed32 rainfallFrequency = math::Fixed32::fromFloat(params.rainfallBelts) / axis;
    const math::Fixed32 coastRange = math::Fixed32::fromFloat(params.coastReach) * axis;

    const core::u32 unreachedDistance = kUnreachedFromSea;
    const Grid<core::u32> seaDistance = distanceToSea(field, seaLevel);

    // ── Rain shadow: carry the highest ground crossed along the wind ─────────
    //
    // Air arriving over a cell has already climbed whatever stood upwind of it.
    // Sweeping against the wind and keeping a decaying maximum of the terrain
    // behind gives, for each cell, how much higher the land upwind rose — which
    // is how much water the air has already dropped before reaching it.
    Heightfield shadow{field.width(), field.depth(), math::Fixed32::zero()};
    const math::Fixed32 shadowStrength = math::Fixed32::fromFloat(params.rainShadow);
    if (shadowStrength.raw() > 0 && span.raw() != 0)
    {
        const core::u32 wind = params.windDirection % 4u;
        const core::i32 windX = kNeighbor4X[wind];
        const core::i32 windZ = kNeighbor4Z[wind];
        // Each cell inherits nine tenths of its upwind neighbour's barrier, so a
        // ridge's influence fades over roughly ten cells rather than forever.
        const math::Fixed32 decay = math::Fixed32::fromFloat(0.9f);

        // Visit cells downwind-last, so a cell's upwind neighbour is already
        // resolved when it is read. Which end each axis starts from is decided by
        // the wind's sign on that axis; the other axis is free.
        for (core::u32 zi = 0u; zi < field.depth(); ++zi)
        {
            const core::u32 z = windZ < 0 ? field.depth() - 1u - zi : zi;
            for (core::u32 xi = 0u; xi < field.width(); ++xi)
            {
                const core::u32 x = windX < 0 ? field.width() - 1u - xi : xi;

                const core::i32 px = static_cast<core::i32>(x) - windX;
                const core::i32 pz = static_cast<core::i32>(z) - windZ;
                math::Fixed32 carried = math::Fixed32::zero();
                if (shadow.contains(px, pz))
                    carried = shadow.at(static_cast<core::u32>(px), static_cast<core::u32>(pz)) * decay;

                const math::Fixed32 here = field.at(x, z);
                shadow.at(x, z) = carried > here ? carried : here;
            }
        }
    }

    const math::Fixed32 maxFlowLog = math::fixedLog2(network.maxAccumulation);

    for (core::u32 z = 0u; z < field.depth(); ++z)
        for (core::u32 x = 0u; x < field.width(); ++x)
        {
            const core::u32 i = field.index(x, z);

            // Baseline rainfall: where it happens to rain, independently of the
            // ground. Mapped from the noise's [-1, 1] into [0, 1].
            const math::Fixed32 rainfallTerm =
                (ValueNoise2D::fbm(math::Fixed32::fromInt(static_cast<core::i32>(x)) * rainfallFrequency,
                                   math::Fixed32::fromInt(static_cast<core::i32>(z)) * rainfallFrequency,
                                   params.rainfallOctaves, params.rainfallSeed) +
                 math::Fixed32::one()) *
                math::Fixed32::half();

            // Upstream drainage on a logarithmic scale. A square root is not enough
            // compression: accumulation spans four orders of magnitude and its root
            // still spans two, so all but the trunk cells land near zero. A logarithm
            // is what the topographic wetness index uses, and it is what makes a
            // modest stream read as wet instead of only the river mouth.
            math::Fixed32 flowTerm = math::Fixed32::zero();
            if (maxFlowLog.raw() > 0)
                flowTerm = math::fixedLog2(network.accumulation[i]) / maxFlowLog;

            const math::Fixed32 altitudeTerm =
                span.raw() != 0 ? math::Fixed32::one() - (field[i] - lowest) / span : math::Fixed32::half();

            math::Fixed32 coastTerm = math::Fixed32::zero();
            if (coastRange.raw() > 0 && seaDistance[i] != unreachedDistance)
            {
                const math::Fixed32 distance = math::Fixed32::fromInt(static_cast<core::i32>(seaDistance[i]));
                if (distance < coastRange)
                    coastTerm = math::Fixed32::one() - distance / coastRange;
            }

            math::Fixed32 wet = rainfallTerm * rainfallWeight + flowTerm * flowWeight + altitudeTerm * altitudeWeight +
                                coastTerm * coastWeight;

            // The shadow subtracts: land behind a high ridge gets what the air had
            // left after climbing it.
            if (shadowStrength.raw() > 0 && span.raw() != 0)
            {
                const math::Fixed32 barrier = (shadow[i] - field[i]) / span;
                if (barrier.raw() > 0)
                    wet = wet - wet * (barrier * shadowStrength);
            }

            if (wet > math::Fixed32::one())
                wet = math::Fixed32::one();
            if (wet < math::Fixed32::zero())
                wet = math::Fixed32::zero();
            moisture[i] = wet;
        }

    // Diffusion widens a channel's influence into its valley: vegetation does
    // not stop at the water's edge.
    smoothHeights(moisture, params.smoothing);

    // ── Rescale so that [0, 1] spans the moisture the LAND actually has ──────
    //
    // The weighted sum above never reaches the ends of its nominal range: the
    // terms rarely peak together, and the ones that do peak together do so under
    // water, where the classifier decides by altitude and never asks about
    // climate. Measured over three map sizes the land moisture topped out between
    // 0.61 and 0.71 and never fell below 0.10, and it drifted with the map's size
    // because the drainage term is normalised against the largest flow — which
    // grows with the map.
    //
    // So absolute thresholds elsewhere would be judging a moving scale, and a
    // "wet" threshold of 0.6 sat above almost the whole land distribution: forest,
    // rainforest and marsh were unreachable not because the classifier was wrong
    // but because nothing could ever satisfy it. Stretching the land's own range
    // onto [0, 1] is the same thing @ref normalizeHeights does for elevation, and
    // for the same reason: a threshold can only mean something against a known
    // range.
    math::Fixed32 landLow = math::Fixed32::max();
    math::Fixed32 landHigh = math::Fixed32::min();
    bool anyLand = false;
    for (core::u32 i = 0u; i < field.cellCount(); ++i)
    {
        if (field[i] <= seaLevel)
            continue;
        anyLand = true;
        if (moisture[i] < landLow)
            landLow = moisture[i];
        if (moisture[i] > landHigh)
            landHigh = moisture[i];
    }

    if (anyLand && landHigh > landLow)
    {
        const math::Fixed32 landSpan = landHigh - landLow;
        for (core::u32 i = 0u; i < moisture.cellCount(); ++i)
        {
            math::Fixed32 scaled = (moisture[i] - landLow) / landSpan;
            if (scaled < math::Fixed32::zero())
                scaled = math::Fixed32::zero();
            if (scaled > math::Fixed32::one())
                scaled = math::Fixed32::one();
            moisture[i] = scaled;
        }
    }
    return moisture;
}

} // namespace lpl::procgen
