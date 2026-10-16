/**
 * @file WorldPartition.cpp
 * @brief WorldPartition implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/container/FlatAtomicHashMap.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>
#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/gpu/IComputeBackend.hpp>

#include <algorithm>
#include <array>
#include <lpl/std/cstring.hpp>
#include <lpl/std/unordered_map.hpp>
#include <lpl/std/vector.hpp>

namespace lpl::ecs {

// ========================================================================== //
//  SpatialCell                                                               //
// ========================================================================== //

/**
 * @brief Lightweight cell stored in the FlatAtomicHashMap.
 *
 * Each cell holds a small vector of entity raw IDs that reside within the
 * corresponding Morton-coded grid cube.
 */
struct SpatialCell {
    core::u64 mortonKey{0};
    pmr::vector<core::u32> entities;

    void insert(core::u32 raw) { entities.push_back(raw); }

    void erase(core::u32 raw)
    {
        auto it = std::find(entities.begin(), entities.end(), raw);
        if (it != entities.end())
        {
            *it = entities.back();
            entities.pop_back();
        }
    }

    [[nodiscard]] bool empty() const noexcept { return entities.empty(); }
};

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct WorldPartition::Impl {
    math::Fixed32 cellSize;
    container::FlatAtomicHashMap<SpatialCell> cells;
    pmr::unordered_map<core::u32, core::u64> entityToMorton;
    gpu::IComputeBackend *gpuBackend{nullptr};

    Impl(math::Fixed32 cs, core::u32 cellCapacity) : cellSize{cs}, cells{cellCapacity} {}
};

/**
 * @brief The grid coordinate, bias included, of the cell @p v falls in along one axis: what
 *        (v / cellSize).toInt() + kMortonBias gives.
 *
 * A Fixed32 division shifts its dividend into 64 bits, which i686 hands to libgcc: three calls a
 * body a tick. For a cell of one unit or more the quotient cannot wrap, and the cell comes from one
 * 32-bit division of the raw values: floor(a / c) for a non-negative a. For a negative a, the
 * quotient rounds toward zero before toInt() drops its fraction downwards, so the cell is
 * -ceil(n / c) when the remainder of n = -a reaches c / 65536, and -floor(n / c) otherwise.
 *
 * @param v A coordinate.
 * @param cellSize The side of a cell, positive.
 */
[[nodiscard]] static inline core::i32 gridAlong(math::Fixed32 v, math::Fixed32 cellSize) noexcept
{
    const core::i32 bias = static_cast<core::i32>(core::kMortonBias);
    if (cellSize.raw() < math::Fixed32::one().raw())
        return (v / cellSize).toInt() + bias;

    const core::u32 side = static_cast<core::u32>(cellSize.raw());
    if (v.raw() >= 0)
        return static_cast<core::i32>(static_cast<core::u32>(v.raw()) / side) + bias;

    const core::u32 below = 0u - static_cast<core::u32>(v.raw());
    const core::u32 whole = below / side;
    const core::u32 rest = below % side;
    const core::u32 cells = whole + (rest >= ((side + 0xFFFFu) >> 16u) ? 1u : 0u);
    return bias - static_cast<core::i32>(cells);
}

/** @brief A 7-bit index with two zero bits after each of its bits: a third of a key's word, per axis. */
static constexpr std::array<core::u32, 128> kSpreadSeven = [] {
    std::array<core::u32, 128> spread{};
    for (core::u32 index = 0u; index < spread.size(); ++index)
        for (core::u32 bit = 0u; bit < 7u; ++bit)
            spread[index] |= ((index >> bit) & 1u) << (3u * bit);
    return spread;
}();

/**
 * @brief The key math::morton::encode3D gives grid coordinates @p x, @p y and @p z, from three
 *        table loads per axis instead of its five rounds of 64-bit shifts and masks.
 *
 * encode3D biases each coordinate again and keeps the low 21 bits of the sum, then sends bit i of
 * x, y and z to bits 3i, 3i + 1 and 3i + 2 of the key. Seven bits of the three axes fill 21 bits
 * of the key, so the key is three such words, at bits 0, 21 and 42: on i686, 32-bit loads, shifts
 * and ors, where each round of encode3D is a pair of them per word.
 */
[[nodiscard]] static core::u64 cellKey(core::i32 x, core::i32 y, core::i32 z) noexcept
{
    const core::u32 bias = static_cast<core::u32>(core::kMortonBias);
    const core::u32 ux = static_cast<core::u32>(x) + bias;
    const core::u32 uy = static_cast<core::u32>(y) + bias;
    const core::u32 uz = static_cast<core::u32>(z) + bias;
    const auto word = [&](core::u32 shift) noexcept -> core::u64 {
        return kSpreadSeven[(ux >> shift) & 0x7Fu] | (kSpreadSeven[(uy >> shift) & 0x7Fu] << 1u) |
               (kSpreadSeven[(uz >> shift) & 0x7Fu] << 2u);
    };
    return word(0u) | (word(7u) << 21u) | (word(14u) << 42u);
}

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

WorldPartition::WorldPartition(math::Fixed32 cellSize, core::u32 cellCapacity)
    : _impl{pmr::make_unique<Impl>(cellSize, cellCapacity)}
{
    LPL_ASSERT(cellSize > math::Fixed32{0});
}

WorldPartition::~WorldPartition() = default;

core::Expected<void> WorldPartition::insertOrUpdate(EntityId id, const math::Vec3<math::Fixed32> &pos)
{
    const core::u64 morton = mortonForPosition(pos);
    const core::u32 raw = id.raw();

    auto it = _impl->entityToMorton.find(raw);
    if (it != _impl->entityToMorton.end())
    {
        if (it->second == morton)
        {
            return {};
        }

        // Remove from old cell
        if (auto *oldCell = _impl->cells.get(it->second))
        {
            oldCell->erase(raw);
        }
    }

    // Insert into new cell (create cell if absent)
    auto *cell = _impl->cells.get(morton);
    if (!cell)
    {
        SpatialCell newCell;
        newCell.mortonKey = morton;
        cell = _impl->cells.insert(morton, std::move(newCell));
        if (!cell)
        {
            return core::makeError(core::ErrorCode::OutOfMemory, "WorldPartition cell pool exhausted");
        }
    }
    cell->insert(raw);
    _impl->entityToMorton[raw] = morton;

    return {};
}

core::Expected<void> WorldPartition::remove(EntityId id)
{
    const core::u32 raw = id.raw();
    auto it = _impl->entityToMorton.find(raw);
    if (it == _impl->entityToMorton.end())
    {
        return core::makeError(core::ErrorCode::NotFound, "Entity not in world partition");
    }

    if (auto *cell = _impl->cells.get(it->second))
    {
        cell->erase(raw);
    }
    _impl->entityToMorton.erase(it);

    return {};
}

void WorldPartition::queryRadius(const math::Vec3<math::Fixed32> &center, math::Fixed32 radius,
                                 pmr::vector<EntityId> &results) const
{
    // Compute the grid-space bounding box for the query sphere
    const core::i32 cx = gridAlong(center.x, _impl->cellSize);
    const core::i32 cy = gridAlong(center.y, _impl->cellSize);
    const core::i32 cz = gridAlong(center.z, _impl->cellSize);

    // Radius in grid cells (ceiling)
    const core::i32 cellRadius = static_cast<core::i32>((radius / _impl->cellSize).toInt()) + 1;

    // Enumerate all cells within the bounding box
    for (core::i32 dx = -cellRadius; dx <= cellRadius; ++dx)
    {
        for (core::i32 dy = -cellRadius; dy <= cellRadius; ++dy)
        {
            for (core::i32 dz = -cellRadius; dz <= cellRadius; ++dz)
            {
                const core::u64 morton = cellKey(cx + dx, cy + dy, cz + dz);

                auto *cell = _impl->cells.get(morton);
                if (!cell)
                {
                    continue;
                }

                for (const core::u32 raw : cell->entities)
                {
                    results.push_back(EntityId{raw});
                }
            }
        }
    }
}

core::u64 WorldPartition::mortonForPosition(const math::Vec3<math::Fixed32> &pos) const noexcept
{
    return cellKey(gridAlong(pos.x, _impl->cellSize), gridAlong(pos.y, _impl->cellSize),
                   gridAlong(pos.z, _impl->cellSize));
}

void WorldPartition::setGpuBackend(gpu::IComputeBackend *backend) noexcept
{
    _impl->gpuBackend = backend;
    if (backend)
    {
        core::Log::info("WorldPartition", "GPU backend registered");
        core::Log::info("WorldPartition", backend->name());
    }
    else
    {
        core::Log::info("WorldPartition", "GPU backend cleared, falling back to CPU");
    }
}

void WorldPartition::step(core::f32 dt)
{
    const core::u32 entityCount = static_cast<core::u32>(_impl->entityToMorton.size());

    if (_impl->gpuBackend && entityCount >= kGpuThreshold)
    {
        // ── GPU path ────────────────────────────────────────────────────────
        // Pack entity raw IDs into a host buffer, upload, dispatch, download.
        // The physics kernel (gpu/src/PhysicsKernel.cu) expects:
        //   args[0..3]  float dt
        //   args[4..7]  u32   entity count
        // followed by the raw entity IDs as u32[entityCount].
        const core::usize payloadBytes =
            sizeof(core::f32) + sizeof(core::u32) + static_cast<core::usize>(entityCount) * sizeof(core::u32);

        pmr::vector<core::byte> argBuf(payloadBytes);
        core::byte *ptr = argBuf.data();

        pmr::memcpy(ptr, &dt, sizeof(dt));
        ptr += sizeof(dt);
        pmr::memcpy(ptr, &entityCount, sizeof(entityCount));
        ptr += sizeof(entityCount);

        for (const auto &[raw, _cellMorton] : _impl->entityToMorton)
        {
            pmr::memcpy(ptr, &raw, sizeof(raw));
            ptr += sizeof(raw);
        }

        constexpr core::u32 kBlockDim = 256;
        const core::u32 gridDim = (entityCount + kBlockDim - 1) / kBlockDim;

        if (auto res =
                _impl->gpuBackend->dispatch("physics_step", gridDim, kBlockDim, std::span<const core::byte>{argBuf});
            !res)
        {
            core::Log::warn("WorldPartition", "GPU dispatch failed, falling back to CPU this tick");
            goto cpu_fallback;
        }

        [[maybe_unused]] auto sync = _impl->gpuBackend->synchronize();
        return;
    }

cpu_fallback:
    // ── CPU path ─────────────────────────────────────────────────────────────
    // Physics integration is delegated to CpuPhysicsBackend through the
    // SystemScheduler (PhysicsSystem / SpatialGrid). WorldPartition::step()
    // serves only as the GPU dispatch gateway; nothing to do here on the
    // CPU path.
    (void) dt;
}

core::u32 WorldPartition::migrateEntities(const pmr::function<math::Vec3<math::Fixed32>(core::u32)> &positionOf)
{
    core::u32 migrated = 0;

    // Iterate all tracked entity→morton mappings
    // Collect entities that need migration
    pmr::vector<std::pair<core::u32, math::Vec3<math::Fixed32>>> toMigrate;

    for (auto &[raw, oldMorton] : _impl->entityToMorton)
    {
        auto pos = positionOf(raw);
        core::u64 newMorton = mortonForPosition(pos);

        if (newMorton != oldMorton)
        {
            toMigrate.push_back({raw, pos});
        }
    }

    // Apply migrations
    for (auto &[raw, pos] : toMigrate)
    {
        auto entityId = EntityId{raw};
        [[maybe_unused]] auto res = insertOrUpdate(entityId, pos);
        ++migrated;
    }

    return migrated;
}

core::u32 WorldPartition::gcEmptyCells()
{
    core::u32 removed = 0;

    // Collect empty cells and remove them from the FlatAtomicHashMap
    pmr::vector<core::u64> emptyKeys;
    _impl->cells.forEach([&](SpatialCell &cell) {
        if (cell.empty())
        {
            emptyKeys.push_back(cell.mortonKey);
        }
    });

    for (core::u64 key : emptyKeys)
    {
        if (_impl->cells.remove(key))
        {
            ++removed;
        }
    }

    return removed;
}

core::u32 WorldPartition::cellCount() const noexcept { return _impl->cells.size(); }

} // namespace lpl::ecs
