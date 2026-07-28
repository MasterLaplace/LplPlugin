/**
 * @file WaveFunctionCollapse.cpp
 * @brief Implementation of the deterministic tiled WFC solver.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/WaveFunctionCollapse.hpp>

#include <lpl/procgen/Random.hpp>

namespace lpl::procgen {

namespace {

/// Index of the direction facing the opposite way (E<->W, S<->N).
constexpr core::u32 oppositeDirection(core::u32 direction) noexcept
{
    // kNeighbor4 order is {E, W, S, N}, so pairs are (0,1) and (2,3).
    return direction ^ 1u;
}

/**
 * @brief Number of set bits in a possibility mask.
 *
 * The SWAR fold rather than the clear-lowest-bit loop: constant twelve
 * operations instead of one iteration per set bit, and no branch. A mask here is
 * usually near-full early in a solve, which is precisely where the loop is at its
 * worst. `__builtin_popcountll` is avoided deliberately — without a popcount
 * instruction it becomes a call into libgcc, which is not linked freestanding.
 */
constexpr core::u32 popcount64(core::u64 mask) noexcept
{
    mask = mask - ((mask >> 1) & 0x5555555555555555ull);
    mask = (mask & 0x3333333333333333ull) + ((mask >> 2) & 0x3333333333333333ull);
    mask = (mask + (mask >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return static_cast<core::u32>((mask * 0x0101010101010101ull) >> 56);
}

/// Index of the lowest set bit.
core::u32 lowestBit(core::u64 mask) noexcept
{
    core::u32 index = 0u;
    while ((mask & 1u) == 0u)
    {
        mask >>= 1;
        ++index;
    }
    return index;
}

/// A cell awaiting propagation.
struct PendingCell {
    core::u32 x;
    core::u32 z;
};

/// A candidate for collapse, keyed by how many options it still has.
struct EntropyEntry {
    core::u32 options;
    core::u32 cell;
};

/**
 * @brief Total order on entropy entries: fewest options first, then lowest index.
 *
 * The index tiebreak is what preserves the documented behaviour exactly. Scanning
 * the grid picked, among all cells tied for fewest options, the first in scan
 * order — which is the lowest flat index. Popping the lexicographic minimum of
 * (options, index) picks the same cell, so swapping the scan for a heap changes
 * the cost and nothing else.
 */
constexpr bool precedes(const EntropyEntry &a, const EntropyEntry &b) noexcept
{
    return a.options != b.options ? a.options < b.options : a.cell < b.cell;
}

/**
 * @class EntropyHeap
 * @brief Binary min-heap of collapse candidates, with lazy deletion.
 *
 * A cell's option count only ever changes by being rewritten, and there is no way
 * to find and update its existing heap entry cheaply. So entries are never
 * updated: a new one is pushed on every change and the outdated ones are
 * discarded when they surface, recognised by their option count no longer matching
 * the cell's. Standard, and it keeps every operation logarithmic.
 */
class EntropyHeap {
public:
    void reserve(core::usize capacity) { _items.reserve(capacity); }
    [[nodiscard]] bool empty() const { return _items.empty(); }
    void clear() { _items.clear(); }

    void push(EntropyEntry entry)
    {
        _items.push_back(entry);
        core::u32 child = static_cast<core::u32>(_items.size()) - 1u;
        while (child != 0u)
        {
            const core::u32 parent = (child - 1u) / 2u;
            if (!precedes(_items[child], _items[parent]))
                break;
            const EntropyEntry swap = _items[parent];
            _items[parent] = _items[child];
            _items[child] = swap;
            child = parent;
        }
    }

    [[nodiscard]] EntropyEntry pop()
    {
        const EntropyEntry top = _items[0];
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
            const EntropyEntry swap = _items[smallest];
            _items[smallest] = _items[node];
            _items[node] = swap;
            node = smallest;
        }
        return top;
    }

private:
    lpl::pmr::vector<EntropyEntry> _items;
};

/**
 * @class Solver
 * @brief One attempt at filling the grid.
 */
class Solver {
public:
    Solver(const TileSet &tiles, const WfcParams &params, core::u32 seed)
        : _tiles(tiles), _params(params), _random(seed), _possible(params.width, params.depth, 0u),
          _options(params.width, params.depth, 0u), _stack()
    {
        const core::usize cells = static_cast<core::usize>(params.width) * params.depth;
        const core::u64 all =
            _tiles.tileCount >= 64u ? ~core::u64{0} : ((core::u64{1} << _tiles.tileCount) - core::u64{1});
        _possible.fill(all);
        _options.fill(_tiles.tileCount);
        _stack.reserve(cells);
        _candidates.reserve(cells * 2u);

        // Every cell starts as a candidate. The heap can only stay complete if it
        // begins complete: a cell absent from it would never be collapsed, and the
        // solve would stop with that cell still superposed.
        if (_tiles.tileCount > 1u)
            for (core::u32 i = 0u; i < static_cast<core::u32>(cells); ++i)
                _candidates.push(EntropyEntry{_tiles.tileCount, i});
    }

    /// Pins the cells @p preset holds, propagating each as a constraint.
    bool applyPreset(const TileGrid &preset)
    {
        for (core::u32 z = 0u; z < _possible.depth(); ++z)
        {
            for (core::u32 x = 0u; x < _possible.width(); ++x)
            {
                const core::u8 tile = preset.at(x, z);
                if (tile == kNoTile || tile >= _tiles.tileCount)
                    continue;
                setOptions(_possible.index(x, z), core::u64{1} << tile);
                push(x, z);
            }
        }
        return propagate();
    }

    /**
     * @brief Runs the collapse loop.
     * @param outLocalRepairs Incremented once per contradiction absorbed locally.
     * @param outContradictions Incremented on every contradiction.
     * @return true when every cell settled on exactly one tile.
     */
    bool run(core::u32 &outLocalRepairs, core::u32 &outContradictions)
    {
        core::u32 repairsLeft = _params.localRepairBudget;

        for (;;)
        {
            core::u32 x = 0u;
            core::u32 z = 0u;
            if (!findLowestEntropy(x, z))
                return true; // every cell is decided

            collapse(x, z);
            if (propagate())
                continue;

            ++outContradictions;

            // Layered recovery: clear a neighbourhood around the failure and
            // let it re-solve against the borders that survived, rather than
            // discarding the whole grid.
            if (_params.localRepairRadius == 0u || repairsLeft == 0u)
                return false;
            --repairsLeft;
            ++outLocalRepairs;
            if (!repairAround(x, z))
                return false;
        }
    }

    /// Writes the settled tiles out.
    TileGrid harvest() const
    {
        TileGrid grid{_possible.width(), _possible.depth(), kNoTile};
        for (core::u32 i = 0u; i < _possible.cellCount(); ++i)
            grid[i] = _possible[i] == 0u ? kNoTile : static_cast<core::u8>(lowestBit(_possible[i]));
        return grid;
    }

private:
    void push(core::u32 x, core::u32 z) { _stack.push_back(PendingCell{x, z}); }

    /// Rewrites a cell's options, keeping the cached count and heap in step.
    void setOptions(core::u32 cell, core::u64 mask)
    {
        _possible[cell] = mask;
        const core::u32 count = popcount64(mask);
        _options[cell] = static_cast<core::u8>(count);
        if (count > 1u)
            _candidates.push(EntropyEntry{count, cell});
    }

    /// Fewest remaining options above 1; ties go to the lowest index.
    bool findLowestEntropy(core::u32 &outX, core::u32 &outZ)
    {
        while (!_candidates.empty())
        {
            const EntropyEntry entry = _candidates.pop();
            // Outdated: the cell has been rewritten since this entry was made, so
            // a newer entry for it is already in the heap.
            if (_options[entry.cell] != static_cast<core::u8>(entry.options) || entry.options <= 1u)
                continue;
            outX = entry.cell % _possible.width();
            outZ = entry.cell / _possible.width();
            // Put it back: a contradiction may send the solve through a local
            // repair that reopens this very cell, and the entry has to survive that.
            _candidates.push(entry);
            return true;
        }
        return false;
    }

    /// Picks one tile from a cell's options, weighted by tile frequency.
    void collapse(core::u32 x, core::u32 z)
    {
        const core::u32 cell = _possible.index(x, z);
        const core::u64 options = _possible[cell];

        core::u32 total = 0u;
        for (core::u64 rest = options; rest != 0u; rest &= rest - 1u)
            total += _tiles.weight[lowestBit(rest)];

        if (total == 0u)
        {
            // Every remaining option has zero weight: fall back to the first so
            // the solve makes progress instead of stalling on a authoring slip.
            setOptions(cell, core::u64{1} << lowestBit(options));
            push(x, z);
            return;
        }

        core::u32 roll = _random.below(total);
        for (core::u64 rest = options; rest != 0u; rest &= rest - 1u)
        {
            const core::u32 tile = lowestBit(rest);
            const core::u32 weight = _tiles.weight[tile];
            if (roll < weight)
            {
                setOptions(cell, core::u64{1} << tile);
                push(x, z);
                return;
            }
            roll -= weight;
        }
    }

    /**
     * @brief Removes now-impossible options from neighbours, transitively.
     * @return false when a cell was left with no option at all.
     */
    bool propagate()
    {
        while (!_stack.empty())
        {
            const PendingCell cell = _stack[_stack.size() - 1u];
            _stack.pop_back();

            const core::u64 options = _possible.at(cell.x, cell.z);
            if (options == 0u)
                return false;

            for (core::u32 direction = 0u; direction < 4u; ++direction)
            {
                const core::i32 nx = static_cast<core::i32>(cell.x) + kNeighbor4X[direction];
                const core::i32 nz = static_cast<core::i32>(cell.z) + kNeighbor4Z[direction];
                if (!_possible.contains(nx, nz))
                    continue;

                // Union of what each still-possible tile here permits there.
                // Walking the set bits rather than every tile index matters late in
                // a solve, which is where propagation does most of its work: a cell
                // down to two options then costs two lookups instead of tileCount.
                core::u64 permitted = 0u;
                for (core::u64 rest = options; rest != 0u; rest &= rest - 1u)
                    permitted |= _tiles.allowed[lowestBit(rest) * 4u + direction];

                const core::u32 neighbourCell = _possible.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                const core::u64 neighbour = _possible[neighbourCell];
                const core::u64 reduced = neighbour & permitted;
                if (reduced == neighbour)
                    continue; // nothing new to say
                if (reduced == 0u)
                    return false;

                setOptions(neighbourCell, reduced);
                push(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
            }
        }
        return true;
    }

    /**
     * @brief Reopens every cell within the repair radius and re-propagates.
     *
     * The cleared cells go back to full superposition; their surviving
     * neighbours immediately constrain them again, which is what makes the
     * repair local rather than a partial restart.
     */
    bool repairAround(core::u32 x, core::u32 z)
    {
        const core::u64 all =
            _tiles.tileCount >= 64u ? ~core::u64{0} : ((core::u64{1} << _tiles.tileCount) - core::u64{1});
        const core::i32 radius = static_cast<core::i32>(_params.localRepairRadius);

        _stack.clear();
        for (core::i32 dz = -radius; dz <= radius; ++dz)
        {
            for (core::i32 dx = -radius; dx <= radius; ++dx)
            {
                const core::i32 cx = static_cast<core::i32>(x) + dx;
                const core::i32 cz = static_cast<core::i32>(z) + dz;
                if (!_possible.contains(cx, cz))
                    continue;
                setOptions(_possible.index(static_cast<core::u32>(cx), static_cast<core::u32>(cz)), all);
            }
        }

        // Re-seed propagation from the ring just outside the cleared area: those
        // cells are still decided and are what the repair must agree with.
        for (core::i32 dz = -radius - 1; dz <= radius + 1; ++dz)
        {
            for (core::i32 dx = -radius - 1; dx <= radius + 1; ++dx)
            {
                if (dx > -radius - 1 && dx < radius + 1 && dz > -radius - 1 && dz < radius + 1)
                    continue; // interior: just cleared
                const core::i32 cx = static_cast<core::i32>(x) + dx;
                const core::i32 cz = static_cast<core::i32>(z) + dz;
                if (_possible.contains(cx, cz))
                    push(static_cast<core::u32>(cx), static_cast<core::u32>(cz));
            }
        }
        return propagate();
    }

    const TileSet &_tiles;
    const WfcParams &_params;
    Random _random;
    Grid<core::u64> _possible;
    /// Cached popcount of every cell's mask, so entropy never recounts bits.
    Grid<core::u8> _options;
    EntropyHeap _candidates;
    lpl::pmr::vector<PendingCell> _stack;
};

} // namespace

void TileSet::reset(core::u32 count)
{
    tileCount = count > kMaxTiles ? kMaxTiles : count;
    allowed.clear();
    weight.clear();
    allowed.resize(static_cast<core::usize>(tileCount) * 4u, core::u64{0});
    weight.resize(tileCount, 1u);
}

void TileSet::allow(core::u32 tile, core::u32 direction, core::u32 neighbour)
{
    if (tile >= tileCount || neighbour >= tileCount || direction >= 4u)
        return;
    allowed[tile * 4u + direction] |= core::u64{1} << neighbour;
    // Adjacency only means anything both ways round.
    allowed[neighbour * 4u + oppositeDirection(direction)] |= core::u64{1} << tile;
}

void TileSet::allowAnywhere(core::u32 a, core::u32 b)
{
    for (core::u32 direction = 0u; direction < 4u; ++direction)
        allow(a, direction, b);
}

void TileSet::setWeight(core::u32 tile, core::u32 value)
{
    if (tile < tileCount)
        weight[tile] = value;
}

bool TileSet::valid() const
{
    if (tileCount == 0u || allowed.size() != static_cast<core::usize>(tileCount) * 4u)
        return false;
    for (core::u32 tile = 0u; tile < tileCount; ++tile)
    {
        // A tile that may sit next to nothing in some direction can never be
        // placed away from that border, which is almost always an authoring
        // slip rather than an intent.
        for (core::u32 direction = 0u; direction < 4u; ++direction)
            if (allowed[tile * 4u + direction] == 0u)
                return false;
    }
    return true;
}

WfcResult solveWfc(const TileSet &tiles, const WfcParams &params, const TileGrid *preset)
{
    WfcResult result;
    if (!tiles.valid() || params.width == 0u || params.depth == 0u)
        return result;

    const core::u32 attempts = params.maxAttempts == 0u ? 1u : params.maxAttempts;
    for (core::u32 attempt = 0u; attempt < attempts; ++attempt)
    {
        ++result.attempts;

        // Each attempt draws from its own stream, so a retry explores a
        // different arrangement instead of replaying the failed one.
        Solver solver{tiles, params, deriveStream(params.seed, 0x5EEDu + attempt).state()};

        bool ok = true;
        if (preset != nullptr && preset->width() == params.width && preset->depth() == params.depth)
            ok = solver.applyPreset(*preset);

        if (ok && solver.run(result.localRepairs, result.contradictions))
        {
            result.solved = true;
            result.tiles = solver.harvest();
            return result;
        }
        result.tiles = solver.harvest();
    }
    return result;
}

core::u32 countAdjacencyViolations(const TileGrid &grid, const TileSet &tiles)
{
    core::u32 violations = 0u;
    for (core::u32 z = 0u; z < grid.depth(); ++z)
    {
        for (core::u32 x = 0u; x < grid.width(); ++x)
        {
            const core::u8 tile = grid.at(x, z);
            if (tile == kNoTile || tile >= tiles.tileCount)
            {
                ++violations;
                continue;
            }
            // Only forward directions, so each pair is judged once.
            for (core::u32 direction = 0u; direction < 4u; direction += 2u)
            {
                const core::i32 nx = static_cast<core::i32>(x) + kNeighbor4X[direction];
                const core::i32 nz = static_cast<core::i32>(z) + kNeighbor4Z[direction];
                if (!grid.contains(nx, nz))
                    continue;
                const core::u8 neighbour = grid.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                if (neighbour == kNoTile || neighbour >= tiles.tileCount)
                    continue;
                if ((tiles.allowed[tile * 4u + direction] & (core::u64{1} << neighbour)) == 0u)
                    ++violations;
            }
        }
    }
    return violations;
}

TileSet makeTerrainTileSet()
{
    TileSet set;
    set.reset(5u);

    // A gradient: each step may touch itself and its immediate neighbours in the
    // sequence, so water never abuts grass without sand between them.
    constexpr core::u32 kSequence[] = {
        static_cast<core::u32>(TerrainTile::Water), static_cast<core::u32>(TerrainTile::Sand),
        static_cast<core::u32>(TerrainTile::Grass), static_cast<core::u32>(TerrainTile::Forest),
        static_cast<core::u32>(TerrainTile::Rock)};
    for (core::u32 i = 0u; i < 5u; ++i)
    {
        set.allowAnywhere(kSequence[i], kSequence[i]);
        if (i + 1u < 5u)
            set.allowAnywhere(kSequence[i], kSequence[i + 1u]);
    }

    // Open water and grass dominate; rock and forest are accents.
    set.setWeight(static_cast<core::u32>(TerrainTile::Water), 4u);
    set.setWeight(static_cast<core::u32>(TerrainTile::Sand), 2u);
    set.setWeight(static_cast<core::u32>(TerrainTile::Grass), 5u);
    set.setWeight(static_cast<core::u32>(TerrainTile::Forest), 3u);
    set.setWeight(static_cast<core::u32>(TerrainTile::Rock), 1u);
    return set;
}

} // namespace lpl::procgen
