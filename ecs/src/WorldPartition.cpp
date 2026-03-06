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
#include <cstring>
#include <unordered_map>
#include <vector>

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
    std::vector<core::u32> entities;

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

/// Legacy used WORLD_CAPACITY = 1 << 16 = 65536 cells.
static constexpr core::u32 kWorldCapacity = 1u << 16u;

struct WorldPartition::Impl {
    math::Fixed32 cellSize;
    container::FlatAtomicHashMap<SpatialCell> cells;
    std::unordered_map<core::u32, core::u64> entityToMorton;
    gpu::IComputeBackend *gpuBackend{nullptr};

    explicit Impl(math::Fixed32 cs) : cellSize{cs}, cells{kWorldCapacity} {}
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

WorldPartition::WorldPartition(math::Fixed32 cellSize) : _impl{std::make_unique<Impl>(cellSize)}
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
                                 std::vector<EntityId> &results) const
{
    // Compute the grid-space bounding box for the query sphere
    const auto toGrid = [&](math::Fixed32 v) -> core::i32 {
        const auto intVal = (v / _impl->cellSize).toInt();
        return intVal + static_cast<core::i32>(core::kMortonBias);
    };

    const core::i32 cx = toGrid(center.x);
    const core::i32 cy = toGrid(center.y);
    const core::i32 cz = toGrid(center.z);

    // Radius in grid cells (ceiling)
    const core::i32 cellRadius = static_cast<core::i32>((radius / _impl->cellSize).toInt()) + 1;

    // Enumerate all cells within the bounding box
    for (core::i32 dx = -cellRadius; dx <= cellRadius; ++dx)
    {
        for (core::i32 dy = -cellRadius; dy <= cellRadius; ++dy)
        {
            for (core::i32 dz = -cellRadius; dz <= cellRadius; ++dz)
            {
                const core::u64 morton = math::morton::encode3D(cx + dx, cy + dy, cz + dz);

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
    const auto toGrid = [&](math::Fixed32 v) -> core::i32 {
        const auto intVal = (v / _impl->cellSize).toInt();
        return intVal + static_cast<core::i32>(core::kMortonBias);
    };

    return math::morton::encode3D(toGrid(pos.x), toGrid(pos.y), toGrid(pos.z));
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

        std::vector<core::byte> argBuf(payloadBytes);
        core::byte *ptr = argBuf.data();

        std::memcpy(ptr, &dt, sizeof(dt));
        ptr += sizeof(dt);
        std::memcpy(ptr, &entityCount, sizeof(entityCount));
        ptr += sizeof(entityCount);

        for (const auto &[raw, _cellMorton] : _impl->entityToMorton)
        {
            std::memcpy(ptr, &raw, sizeof(raw));
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

core::u32 WorldPartition::migrateEntities(const std::function<math::Vec3<math::Fixed32>(core::u32)> &positionOf)
{
    core::u32 migrated = 0;

    // Iterate all tracked entity→morton mappings
    // Collect entities that need migration
    std::vector<std::pair<core::u32, math::Vec3<math::Fixed32>>> toMigrate;

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
    std::vector<core::u64> emptyKeys;
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
