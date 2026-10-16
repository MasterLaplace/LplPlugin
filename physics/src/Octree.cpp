/**
 * @file Octree.cpp
 * @brief Morton-sorted flat octree with LSD radix sort (4-pass).
 *
 * Ported from legacy FlatDynamicOctree.hpp:
 *   - LSD radix sort: 4 passes × 16-bit buckets = 64-bit Morton keys
 *   - Flat node array: cache-friendly, GPU-transferable POD layout
 *   - Recursive octant build: splits sorted refs by Morton octant bits
 *   - AABB query: recursive tree traversal with early rejection
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/memory/PinnedAllocator.hpp>
#include <lpl/physics/Octree.hpp>
#include <lpl/std/memory.hpp>
#include <lpl/std/vector.hpp>

#include <algorithm>
#include <array>
#include <bit>

namespace lpl::physics {

// ========================================================================== //
//  Impl — Flat node tree with LSD radix sort                                 //
// ========================================================================== //

struct Octree::Impl {
    static constexpr core::u8 kMaxDepth = 8;
    core::u32 leafCapacity{32u};

    /** @brief A box that holds nothing and intersects nothing: the reach of an empty node. */
    static constexpr math::AABB<math::Fixed32> kNothing{
        {math::Fixed32::fromRaw(0x7FFFFFFF),      math::Fixed32::fromRaw(0x7FFFFFFF),      math::Fixed32::fromRaw(0x7FFFFFFF)},
        {math::Fixed32::fromRaw(-0x7FFFFFFF - 1), math::Fixed32::fromRaw(-0x7FFFFFFF - 1),
         math::Fixed32::fromRaw(-0x7FFFFFFF - 1)                                                                             }
    };

    /** @brief Flat node (POD, GPU-transferable). */
    struct FlatNode {
        math::AABB<math::Fixed32> bound; /**< The cell: the octant of its parent this node splits. */
        /**
         * Every box the node and its descendants hold, merged. An entry belongs to the cell its
         * centre falls in, and its box reaches past the cell by up to half its size, so a region can
         * intersect an entry without intersecting its cell: a search rejects on the reach, never on
         * the cell.
         */
        math::AABB<math::Fixed32> reach{kNothing};
        core::i32 firstChild{-1};
        core::u32 entityStart{0};
        core::u32 entityCount{0};
    };

    /** @brief Entity reference sorted by Morton key. */
    struct EntityEntry {
        core::u32 objectId;
        core::u32 morton; /**< kKeyBits bits: a word, so the sort and the split shift one register. */
        math::AABB<math::Fixed32> aabb;
    };

    using PinnedNode = memory::PinnedAllocator<FlatNode>;
    using PinnedEntry = memory::PinnedAllocator<EntityEntry>;

    math::AABB<math::Fixed32> worldBounds;
    lpl::pmr::vector<FlatNode, PinnedNode> nodes;
    lpl::pmr::vector<EntityEntry, PinnedEntry> sortedEntries;
    lpl::pmr::vector<EntityEntry, PinnedEntry> tempEntries;
    /**
     * Object id → entry slot, as a dense array rather than a hash map.
     *
     * A node-based map allocates once per insert, and a broad-phase refills the
     * index every step — that was 1024 allocations per tick on the cube-pile
     * sample, inside the authoritative step. A flat array indexed by the object
     * id costs one u32 per id and never touches the allocator once warm, which
     * is what the real-time contract asks for.
     *
     * Trade-off, stated because it is one: the array is sized by the LARGEST
     * object id, not by the object count, so an index keyed on a sparse id space
     * spends 4 bytes per unused id. Every caller feeds it dense ids (a chunk's
     * entity slots), which is the case this structure is built for.
     */
    lpl::pmr::vector<core::u32> idToIndex;
    bool dirty{false};
    std::array<core::u32, 3> keyShifts{};    /**< Per axis, the right shift from an offset to its cell. */
    std::array<core::u32, 3> keyLastCells{}; /**< Per axis, the last cell; 0 along an axis with no extent. */

    static constexpr core::u32 kNoIndex = 0xFFFFFFFFu;

    Impl(const math::AABB<math::Fixed32> &wb, core::u32 leaf) : leafCapacity{leaf == 0u ? 1u : leaf}, worldBounds{wb}
    {
        fitKeys();
    }

    void setIndex(core::u32 objectId, core::u32 slot)
    {
        if (objectId >= idToIndex.size())
            idToIndex.resize(objectId + 1u, kNoIndex);
        idToIndex[objectId] = slot;
    }

    [[nodiscard]] core::u32 indexOf(core::u32 objectId) const
    {
        return (objectId < idToIndex.size()) ? idToIndex[objectId] : kNoIndex;
    }

    // ────────────────────────────────────────────────────────────────────── //
    //  Morton computation                                                    //
    // ────────────────────────────────────────────────────────────────────── //

    /** @brief Bits of the key per axis: one per level the build can split, and no more. */
    static constexpr core::u32 kGridBits = kMaxDepth;

    /** @brief Bits of a whole key, three axes interleaved. */
    static constexpr core::u32 kKeyBits = 3u * kGridBits;
    static_assert(kKeyBits <= 32u, "a key is one 32-bit word");

    /** @brief Passes of the radix sort, eight bits a pass. */
    static constexpr core::u32 kRadixPasses = (kKeyBits + 7u) / 8u;

    /**
     * @brief A cell index of kGridBits bits with two zero bits after each: the three axes of a key
     *        interleave as spread(x) | spread(y) << 1 | spread(z) << 2, three loads instead of the
     *        64-bit shifts and masks of part1by2, which i686 runs as pairs of 32-bit operations.
     */
    static constexpr std::array<core::u32, (1u << kGridBits)> kSpread = [] {
        std::array<core::u32, (1u << kGridBits)> spread{};
        for (core::u32 index = 0u; index < spread.size(); ++index)
            for (core::u32 bit = 0u; bit < kGridBits; ++bit)
                spread[index] |= ((index >> bit) & 1u) << (3u * bit);
        return spread;
    }();

    /**
     * @brief The centre of an AABB, as a Morton key over THIS TREE'S bounds.
     *
     * Each axis is cut into cells of a power-of-two width, the smallest that fits the tree's extent
     * in 2^kGridBits cells, so a cell index is a shift of the offset from the tree's minimum: no
     * division, which on i686 is a call into libgcc, three of them per body per rebuild before. A key
     * carries one bit per axis per level @ref recurseBuild can split, which is all it reads, so the
     * radix sort runs @ref kRadixPasses passes instead of eight.
     *
     * Where an entry's key places it is a matter of balance, not of correctness: a search rejects on
     * a node's reach, which holds every box beneath it whatever cell it was filed under.
     */
    [[nodiscard]] core::u32 computeMorton(const math::AABB<math::Fixed32> &aabb) const
    {
        const auto c = aabb.center();
        const auto cell = [this](math::Fixed32 v, math::Fixed32 lo, core::u32 axis) -> core::u32 {
            if (v.raw() <= lo.raw())
                return 0u;
            const core::u32 offset = static_cast<core::u32>(v.raw()) - static_cast<core::u32>(lo.raw());
            const core::u32 index = offset >> keyShifts[axis];
            return index < keyLastCells[axis] ? index : keyLastCells[axis];
        };
        return kSpread[cell(c.x, worldBounds.min.x, 0u)] | (kSpread[cell(c.y, worldBounds.min.y, 1u)] << 1u) |
               (kSpread[cell(c.z, worldBounds.min.z, 2u)] << 2u);
    }

    /**
     * @brief Derives, for each axis, the shift that turns an offset into a cell and the last cell,
     *        from the tree's bounds: once when they change, not three times a key.
     */
    void fitKeys() noexcept
    {
        const math::Fixed32 lows[3] = {worldBounds.min.x, worldBounds.min.y, worldBounds.min.z};
        const math::Fixed32 highs[3] = {worldBounds.max.x, worldBounds.max.y, worldBounds.max.z};

        for (core::u32 axis = 0u; axis < 3u; ++axis)
        {
            const core::i64 extent =
                static_cast<core::i64>(highs[axis].raw()) - static_cast<core::i64>(lows[axis].raw());
            const core::u32 width =
                extent > 0 ? static_cast<core::u32>(std::bit_width(static_cast<core::u64>(extent))) : 0u;
            keyShifts[axis] = width > kGridBits ? width - kGridBits : 0u;
            keyLastCells[axis] = extent > 0 ? (1u << kGridBits) - 1u : 0u;
        }
    }

    // ────────────────────────────────────────────────────────────────────── //
    //  LSD Radix Sort — kRadixPasses passes × 8-bit buckets                  //
    // ────────────────────────────────────────────────────────────────────── //

    /**
     * @brief Sorts sortedEntries by morton key using LSD radix sort, stable.
     *
     * One pass per 8-bit digit of the key: count, prefix-sum offsets, scatter into the other buffer.
     * An 8-bit digit keeps the histograms at 256 entries, 2 KiB of stack: a 16-bit digit took 512 KiB
     * and overran the kernel stack. After an odd number of passes the sorted entries sit in the other
     * buffer, and the two vectors swap, which moves no entry and allocates nothing.
     *
     * Complexity: O(kRadixPasses × n).
     */
    void radixSort()
    {
        const core::u32 count = static_cast<core::u32>(sortedEntries.size());
        if (count <= 1)
        {
            return;
        }

        if (tempEntries.size() < sortedEntries.size())
        {
            tempEntries.resize(sortedEntries.size());
        }

        EntityEntry *src = sortedEntries.data();
        EntityEntry *dst = tempEntries.data();

        for (core::u32 pass = 0; pass < kRadixPasses; ++pass)
        {
            core::u32 counts[256] = {};
            const core::u32 shift = pass * 8u;

            for (core::u32 i = 0; i < count; ++i)
            {
                const core::u8 bucket = (src[i].morton >> shift) & 0xFFu;
                ++counts[bucket];
            }

            core::u32 offsets[256];
            offsets[0] = 0;
            for (core::u32 i = 1; i < 256u; ++i)
            {
                offsets[i] = offsets[i - 1] + counts[i - 1];
            }

            for (core::u32 i = 0; i < count; ++i)
            {
                const core::u8 bucket = (src[i].morton >> shift) & 0xFFu;
                dst[offsets[bucket]++] = src[i];
            }

            std::swap(src, dst);
        }

        if constexpr (kRadixPasses % 2u != 0u)
        {
            sortedEntries.swap(tempEntries);
        }
    }

    // ────────────────────────────────────────────────────────────────────── //
    //  Recursive tree build                                                  //
    // ────────────────────────────────────────────────────────────────────── //

    /**
     * @brief Recursively subdivides sorted entities into octree nodes.
     *
     * Uses the Morton key bits at each depth level to determine which
     * octant each entity belongs to. Since entities are pre-sorted by
     * Morton key, each octant forms a contiguous range.
     */
    void recurseBuild(core::u32 nodeIdx, core::u32 start, core::u32 end, core::u8 depth)
    {
        const core::u32 entityCount = end - start;

        if (entityCount <= leafCapacity || depth >= kMaxDepth)
        {
            auto &node = nodes[nodeIdx];
            node.entityStart = start;
            node.entityCount = entityCount;
            node.reach = kNothing;
            for (core::u32 index = start; index < end; ++index)
                node.reach = node.reach.merge(sortedEntries[index].aabb);
            return;
        }

        // Allocate 8 children
        const core::u32 firstChildIdx = static_cast<core::u32>(nodes.size());
        nodes.resize(firstChildIdx + 8);
        nodes[nodeIdx].firstChild = static_cast<core::i32>(firstChildIdx);
        // An interior node holds NOTHING: its objects now live in its children. The
        // root was seeded with the whole set before this walk, and leaving that count
        // in place made every traversal visit each object twice — once at the root
        // and once in its leaf. Harmless-looking in a broad-phase, where a duplicate
        // pair is merely redundant work; fatal to a cull, which returns the payload
        // to the caller and doubled the visible set.
        nodes[nodeIdx].entityStart = start;
        nodes[nodeIdx].entityCount = 0u;

        const auto &parentBound = nodes[nodeIdx].bound;
        const auto center = parentBound.center();
        const auto mn = parentBound.min;
        const auto mx = parentBound.max;

        // Compute child bounds and split entities by octant
        const core::u32 shift = (kGridBits - 1u - depth) * 3u;
        core::u32 current = start;

        for (core::u8 octant = 0; octant < 8; ++octant)
        {
            // Set child bounds based on octant bits
            math::Vec3<math::Fixed32> childMin, childMax;

            // X axis (bit 0)
            if (octant & 1)
            {
                childMin.x = center.x;
                childMax.x = mx.x;
            }
            else
            {
                childMin.x = mn.x;
                childMax.x = center.x;
            }

            // Y axis (bit 1)
            if (octant & 2)
            {
                childMin.y = center.y;
                childMax.y = mx.y;
            }
            else
            {
                childMin.y = mn.y;
                childMax.y = center.y;
            }

            // Z axis (bit 2)
            if (octant & 4)
            {
                childMin.z = center.z;
                childMax.z = mx.z;
            }
            else
            {
                childMin.z = mn.z;
                childMax.z = center.z;
            }

            nodes[firstChildIdx + octant].bound = {childMin, childMax};

            // Find contiguous range for this octant (sorted by Morton)
            const core::u32 childStart = current;
            while (current < end)
            {
                const core::u8 entOctant = static_cast<core::u8>((sortedEntries[current].morton >> shift) & 7u);
                if (entOctant != octant)
                {
                    break;
                }
                ++current;
            }

            recurseBuild(firstChildIdx + octant, childStart, current, depth + 1);
        }

        math::AABB<math::Fixed32> reach = kNothing;
        for (core::u32 octant = 0; octant < 8; ++octant)
            reach = reach.merge(nodes[firstChildIdx + octant].reach);
        nodes[nodeIdx].reach = reach;
    }

    // ────────────────────────────────────────────────────────────────────── //
    //  Recursive AABB query                                                  //
    // ────────────────────────────────────────────────────────────────────── //

    /// @copydoc Octree::queryVisible
    void visibleRecurse(core::u32 nodeIdx,
                        const lpl::pmr::function<bool(const math::AABB<math::Fixed32> &)> &nodeVisible,
                        const lpl::pmr::function<void(core::u32)> &callback, core::u32 *visited,
                        core::u32 *pruned) const
    {
        const auto &node = nodes[nodeIdx];
        if (visited != nullptr)
            ++(*visited);
        if (!nodeVisible(node.reach))
        {
            if (pruned != nullptr)
                ++(*pruned);
            return;
        }

        for (core::u32 i = 0; i < node.entityCount; ++i)
        {
            const auto &entry = sortedEntries[node.entityStart + i];
            if (nodeVisible(entry.aabb))
            {
                callback(entry.objectId);
            }
        }

        if (node.firstChild >= 0)
        {
            const core::u32 fc = static_cast<core::u32>(node.firstChild);
            for (core::u32 i = 0; i < 8; ++i)
            {
                if (nodes[fc + i].entityCount != 0u || nodes[fc + i].firstChild >= 0)
                    visibleRecurse(fc + i, nodeVisible, callback, visited, pruned);
            }
        }
    }

    /** @brief Nodes a search can hold pending: up to seven siblings wait at each level, plus the root. */
    static constexpr core::u32 kPendingNodes = 7u * kMaxDepth + 1u;

    /**
     * @brief Hands @p visitor the entries numbered @p firstId or above that intersect @p region, of
     *        the root and its descendants, the root's reach having been found to intersect it: depth
     *        first, children in octant order, each child tested on its reach before it is visited.
     *
     * A stack of pending nodes stands in for a call per node. Each node's intersecting children go
     * on it last octant first, so the first comes off first and its subtree is done before the next:
     * the order of the recursion it replaces, without a frame per node.
     *
     * The node's fields and the entries' address are read once: an append through @p visitor
     * stores a 32-bit word that the compiler cannot prove misses them, so read through the node they
     * were read again after every append.
     */
    template <typename Visit>
    void visitTree(const math::AABB<math::Fixed32> &region, core::u32 firstId, Visit &visitor) const
    {
        core::u32 pending[kPendingNodes];
        core::u32 waiting = 0u;
        pending[waiting++] = 0u;

        while (waiting != 0u)
        {
            const FlatNode &node = nodes[pending[--waiting]];
            const EntityEntry *const entries = sortedEntries.data() + node.entityStart;
            const core::u32 entityCount = node.entityCount;
            const core::i32 firstChild = node.firstChild;

            for (core::u32 i = 0; i < entityCount; ++i)
            {
                if (entries[i].objectId >= firstId && entries[i].aabb.intersects(region))
                {
                    visitor(entries[i].objectId);
                }
            }

            if (firstChild < 0)
            {
                continue;
            }
            const FlatNode *const children = nodes.data() + firstChild;
            for (core::u32 octant = 8u; octant-- != 0u;)
            {
                if (children[octant].reach.intersects(region))
                    pending[waiting++] = static_cast<core::u32>(firstChild) + octant;
            }
        }
    }

    /**
     * @brief Hands @p visitor every entry numbered @p firstId or above that intersects @p region, in
     *        the order of @ref visitTree.
     */
    template <typename Visit>
    void visit(const math::AABB<math::Fixed32> &region, core::u32 firstId, Visit &visitor) const
    {
        if (nodes.empty())
        {
            for (const auto &entry : sortedEntries)
            {
                if (entry.objectId >= firstId && entry.aabb.intersects(region))
                {
                    visitor(entry.objectId);
                }
            }
            return;
        }

        if (nodes[0].reach.intersects(region))
        {
            visitTree(region, firstId, visitor);
        }
    }
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

Octree::Octree(const math::AABB<math::Fixed32> &worldBounds, core::u32 leafCapacity)
    : _impl{lpl::pmr::make_unique<Impl>(worldBounds, leafCapacity)}
{
}

Octree::~Octree() = default;

void Octree::insert(core::u32 objectId, const math::AABB<math::Fixed32> &aabb)
{
    const core::u32 idx = static_cast<core::u32>(_impl->sortedEntries.size());
    _impl->sortedEntries.push_back({objectId, _impl->computeMorton(aabb), aabb});
    _impl->setIndex(objectId, idx);
    _impl->dirty = true;
}

void Octree::update(core::u32 objectId, const math::AABB<math::Fixed32> &aabb)
{
    const core::u32 slot = _impl->indexOf(objectId);
    LPL_ASSERT(slot != Impl::kNoIndex);
    auto &entry = _impl->sortedEntries[slot];
    entry.aabb = aabb;
    entry.morton = _impl->computeMorton(aabb);
    _impl->dirty = true;
}

void Octree::remove(core::u32 objectId)
{
    const core::u32 idx = _impl->indexOf(objectId);
    LPL_ASSERT(idx != Impl::kNoIndex);

    const core::u32 last = static_cast<core::u32>(_impl->sortedEntries.size()) - 1;

    if (idx != last)
    {
        _impl->sortedEntries[idx] = _impl->sortedEntries[last];
        _impl->setIndex(_impl->sortedEntries[idx].objectId, idx);
    }
    _impl->sortedEntries.pop_back();
    _impl->idToIndex[objectId] = Impl::kNoIndex;
    _impl->dirty = true;
}

void Octree::query(const math::AABB<math::Fixed32> &region, const lpl::pmr::function<void(core::u32)> &callback) const
{
    _impl->visit(region, 0u, callback);
}

void Octree::collect(const math::AABB<math::Fixed32> &region, core::u32 firstId, lpl::pmr::vector<core::u32> &out) const
{
    auto append = [&out](core::u32 id) { out.push_back(id); };
    _impl->visit(region, firstId, append);
}

core::u32 Octree::keyOf(const math::AABB<math::Fixed32> &aabb) const noexcept { return _impl->computeMorton(aabb); }

void Octree::queryVisible(const lpl::pmr::function<bool(const math::AABB<math::Fixed32> &)> &nodeVisible,
                          const lpl::pmr::function<void(core::u32)> &callback, core::u32 *outNodesVisited,
                          core::u32 *outNodesPruned) const
{
    if (outNodesVisited != nullptr)
        *outNodesVisited = 0u;
    if (outNodesPruned != nullptr)
        *outNodesPruned = 0u;

    if (_impl->nodes.empty())
    {
        // Same fallback as the box query: before the first rebuild there is no
        // hierarchy to descend, and a linear pass is still a correct answer.
        for (const auto &entry : _impl->sortedEntries)
            if (nodeVisible(entry.aabb))
                callback(entry.objectId);
        return;
    }

    _impl->visibleRecurse(0, nodeVisible, callback, outNodesVisited, outNodesPruned);
}

void Octree::rebuild()
{
    if (!_impl->dirty)
    {
        return;
    }

    // Step 1: LSD radix sort by Morton key (O(n))
    _impl->radixSort();

    // Step 2: Rebuild index map after sort
    for (core::u32 i = 0; i < static_cast<core::u32>(_impl->sortedEntries.size()); ++i)
    {
        _impl->setIndex(_impl->sortedEntries[i].objectId, i);
    }

    // Step 3: Build flat node tree.
    //
    // clear() keeps the capacity, but the node count of an octree exceeds its
    // entity count (interior nodes are subdivided 8 at a time), so reserving
    // only entityCount left resize() growing — and growing means calling the
    // allocator from inside an authoritative tick, which the freestanding
    // REAL_TIME mode forbids. Reserve a bound that covers the subdivision so
    // the growth happens once, here, and never on a later tick.
    //
    // Bound: a node either holds entities or splits into 8. With N entities the
    // tree has at most N leaves, hence at most (N - 1) / 7 interior nodes, so
    // (N * 9) / 7 + 8 is a safe ceiling; the +8 covers the root's own split.
    _impl->nodes.clear();
    {
        const auto entityCount = _impl->sortedEntries.size();
        const auto nodeCeiling = (entityCount * 9u) / 7u + 8u;
        if (_impl->nodes.capacity() < nodeCeiling)
            _impl->nodes.reserve(nodeCeiling);
    }
    _impl->nodes.emplace_back();
    _impl->nodes[0].bound = _impl->worldBounds;
    _impl->nodes[0].entityStart = 0;
    _impl->nodes[0].entityCount = static_cast<core::u32>(_impl->sortedEntries.size());

    if (!_impl->sortedEntries.empty())
    {
        _impl->recurseBuild(0, 0, static_cast<core::u32>(_impl->sortedEntries.size()), 0);
    }

    _impl->dirty = false;
}

void Octree::clear() noexcept
{
    _impl->sortedEntries.clear();
    _impl->nodes.clear();
    for (auto &slot : _impl->idToIndex)
        slot = Impl::kNoIndex;
    _impl->dirty = false;
}

void Octree::setWorldBounds(const math::AABB<math::Fixed32> &worldBounds) noexcept
{
    _impl->worldBounds = worldBounds;
    _impl->fitKeys();
    _impl->dirty = true;
}

const math::AABB<math::Fixed32> &Octree::worldBounds() const noexcept { return _impl->worldBounds; }

core::u32 Octree::count() const noexcept { return static_cast<core::u32>(_impl->sortedEntries.size()); }

} // namespace lpl::physics
