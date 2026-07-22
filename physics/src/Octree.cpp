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
#include <lpl/math/Morton.hpp>
#include <lpl/memory/PinnedAllocator.hpp>
#include <lpl/physics/Octree.hpp>
#include <lpl/std/memory.hpp>
#include <lpl/std/vector.hpp>

#include <algorithm>

namespace lpl::physics {

// ========================================================================== //
//  Impl — Flat node tree with LSD radix sort                                 //
// ========================================================================== //

struct Octree::Impl {
    static constexpr core::u8 kMaxDepth = 8;
    static constexpr core::u32 kLeafCapacity = 32;

    /** @brief Flat node (POD, GPU-transferable). */
    struct FlatNode {
        math::AABB<math::Fixed32> bound;
        core::i32 firstChild{-1};
        core::u32 entityStart{0};
        core::u32 entityCount{0};
    };

    /** @brief Entity reference sorted by Morton key. */
    struct EntityEntry {
        core::u32 objectId;
        core::u64 morton;
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

    static constexpr core::u32 kNoIndex = 0xFFFFFFFFu;

    explicit Impl(const math::AABB<math::Fixed32> &wb) : worldBounds{wb} {}

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

    [[nodiscard]] core::u64 computeMorton(const math::AABB<math::Fixed32> &aabb) const
    {
        const auto c = aabb.center();
        const auto toGrid = [](math::Fixed32 v) -> core::i32 {
            return v.toInt() + static_cast<core::i32>(core::kMortonBias);
        };
        return math::morton::encode3D(toGrid(c.x), toGrid(c.y), toGrid(c.z));
    }

    // ────────────────────────────────────────────────────────────────────── //
    //  LSD Radix Sort — 4 passes × 16-bit buckets                           //
    // ────────────────────────────────────────────────────────────────────── //

    /**
     * @brief Sorts sortedEntries by morton key using LSD radix sort.
     *
     * 4 passes over 16-bit digits of the 64-bit Morton key.
     * Each pass: count → prefix-sum offsets → scatter into temp buffer → swap.
     *
     * Complexity: O(4 × n) = O(n) linear in entity count.
     */
    void radixSort()
    {
        const core::u32 count = static_cast<core::u32>(sortedEntries.size());
        if (count <= 1)
        {
            return;
        }

        // Grown here, once, rather than left to grow during a later tick.
        if (tempEntries.size() < sortedEntries.size())
        {
            tempEntries.resize(sortedEntries.size());
        }

        EntityEntry *src = sortedEntries.data();
        EntityEntry *dst = tempEntries.data();

        // 8-bit radix / 8 passes over the 64-bit Morton key. An 8-bit digit keeps
        // the histograms at 256 entries (2 KiB of stack total) instead of 65536
        // (512 KiB) — the wide version blows a freestanding kernel stack, corrupting
        // adjacent globals, while a hosted stack merely hid the bug. The sorted order
        // is identical either way (LSD radix is digit-width agnostic), so parity
        // signatures are unchanged. 8 passes is even, so the result lands back in the
        // original sortedEntries buffer, as the caller relies on.
        for (core::u8 pass = 0; pass < 8; ++pass)
        {
            // Histogram: count occurrences per 8-bit bucket
            core::u32 counts[256] = {};
            const core::u64 shift = static_cast<core::u64>(pass) * 8u;

            for (core::u32 i = 0; i < count; ++i)
            {
                const core::u8 bucket = (src[i].morton >> shift) & 0xFFu;
                ++counts[bucket];
            }

            // Prefix sum → offsets
            core::u32 offsets[256];
            offsets[0] = 0;
            for (core::u32 i = 1; i < 256u; ++i)
            {
                offsets[i] = offsets[i - 1] + counts[i - 1];
            }

            // Scatter into destination
            for (core::u32 i = 0; i < count; ++i)
            {
                const core::u8 bucket = (src[i].morton >> shift) & 0xFFu;
                dst[offsets[bucket]++] = src[i];
            }

            // Swap src/dst for next pass
            std::swap(src, dst);
        }

        // After 8 passes (even), result is in the original sortedEntries buffer
        // (src == sortedEntries.data() after 8 swaps)
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

        if (entityCount <= kLeafCapacity || depth >= kMaxDepth)
        {
            auto &node = nodes[nodeIdx];
            node.entityStart = start;
            node.entityCount = entityCount;
            return;
        }

        // Allocate 8 children
        const core::u32 firstChildIdx = static_cast<core::u32>(nodes.size());
        nodes.resize(firstChildIdx + 8);
        nodes[nodeIdx].firstChild = static_cast<core::i32>(firstChildIdx);

        const auto &parentBound = nodes[nodeIdx].bound;
        const auto center = parentBound.center();
        const auto mn = parentBound.min;
        const auto mx = parentBound.max;

        // Compute child bounds and split entities by octant
        const core::u64 shift = (20u - depth) * 3u;
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
    }

    // ────────────────────────────────────────────────────────────────────── //
    //  Recursive AABB query                                                  //
    // ────────────────────────────────────────────────────────────────────── //

    void queryRecurse(core::u32 nodeIdx, const math::AABB<math::Fixed32> &region,
                      const lpl::pmr::function<void(core::u32)> &callback) const
    {
        const auto &node = nodes[nodeIdx];

        // Early rejection: region doesn't intersect this node's bounds
        if (!node.bound.intersects(region))
        {
            return;
        }

        // Check leaf entities
        for (core::u32 i = 0; i < node.entityCount; ++i)
        {
            const auto &entry = sortedEntries[node.entityStart + i];
            if (entry.aabb.intersects(region))
            {
                callback(entry.objectId);
            }
        }

        // Recurse into children
        if (node.firstChild >= 0)
        {
            const core::u32 fc = static_cast<core::u32>(node.firstChild);
            for (core::u32 i = 0; i < 8; ++i)
            {
                queryRecurse(fc + i, region, callback);
            }
        }
    }
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

Octree::Octree(const math::AABB<math::Fixed32> &worldBounds) : _impl{lpl::pmr::make_unique<Impl>(worldBounds)} {}

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
    if (_impl->nodes.empty())
    {
        // Fallback: linear scan if tree not built yet
        for (const auto &entry : _impl->sortedEntries)
        {
            if (entry.aabb.intersects(region))
            {
                callback(entry.objectId);
            }
        }
        return;
    }

    _impl->queryRecurse(0, region, callback);
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

core::u32 Octree::count() const noexcept { return static_cast<core::u32>(_impl->sortedEntries.size()); }

} // namespace lpl::physics
