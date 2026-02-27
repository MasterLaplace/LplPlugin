/**
 * @file WorldPartition.cpp
 * @brief WorldPartition implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/ecs/WorldPartition.hpp>
#include <lpl/core/Assert.hpp>

#include <unordered_map>
#include <unordered_set>

namespace lpl::ecs {

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct WorldPartition::Impl
{
    math::Fixed32                                                cellSize;
    std::unordered_map<core::u64, std::unordered_set<core::u32>> cells;
    std::unordered_map<core::u32, core::u64>                     entityToMorton;

    explicit Impl(math::Fixed32 cs) : cellSize{cs} {}
};

// ========================================================================== //
//  Public API                                                                //
// ========================================================================== //

WorldPartition::WorldPartition(math::Fixed32 cellSize)
    : _impl{std::make_unique<Impl>(cellSize)}
{
    LPL_ASSERT(cellSize > math::Fixed32{0});
}

WorldPartition::~WorldPartition() = default;

core::Expected<void> WorldPartition::insertOrUpdate(
    EntityId id,
    const math::Vec3<math::Fixed32>& pos)
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

        _impl->cells[it->second].erase(raw);
        if (_impl->cells[it->second].empty())
        {
            _impl->cells.erase(it->second);
        }
    }

    _impl->cells[morton].insert(raw);
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

    _impl->cells[it->second].erase(raw);
    if (_impl->cells[it->second].empty())
    {
        _impl->cells.erase(it->second);
    }
    _impl->entityToMorton.erase(it);

    return {};
}

void WorldPartition::queryRadius(
    const math::Vec3<math::Fixed32>& center,
    math::Fixed32 radius,
    std::vector<EntityId>& results) const
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
    const core::i32 cellRadius = static_cast<core::i32>(
        (radius / _impl->cellSize).toInt()) + 1;

    // Enumerate all cells within the bounding box
    for (core::i32 dx = -cellRadius; dx <= cellRadius; ++dx)
    {
        for (core::i32 dy = -cellRadius; dy <= cellRadius; ++dy)
        {
            for (core::i32 dz = -cellRadius; dz <= cellRadius; ++dz)
            {
                const core::u64 morton = math::morton::encode3D(
                    cx + dx, cy + dy, cz + dz);

                auto it = _impl->cells.find(morton);
                if (it == _impl->cells.end())
                {
                    continue;
                }

                for (const core::u32 raw : it->second)
                {
                    results.push_back(EntityId{raw});
                }
            }
        }
    }
}

core::u64 WorldPartition::mortonForPosition(const math::Vec3<math::Fixed32>& pos) const noexcept
{
    const auto toGrid = [&](math::Fixed32 v) -> core::i32 {
        const auto intVal = (v / _impl->cellSize).toInt();
        return intVal + static_cast<core::i32>(core::kMortonBias);
    };

    return math::morton::encode3D(toGrid(pos.x), toGrid(pos.y), toGrid(pos.z));
}

void WorldPartition::step(core::f32 /*dt*/)
{
    // TODO: auto-select GPU backend when entity count > kGpuThreshold
    // For now, CPU-only — physics handled by CpuPhysicsBackend through SystemScheduler
    // This method serves as the integration point for legacy stepCPU/stepGPU dispatch
}

core::u32 WorldPartition::gcEmptyCells()
{
    core::u32 removed = 0;

    for (auto it = _impl->cells.begin(); it != _impl->cells.end(); )
    {
        if (it->second.empty())
        {
            it = _impl->cells.erase(it);
            ++removed;
        }
        else
        {
            ++it;
        }
    }

    return removed;
}

core::u32 WorldPartition::cellCount() const noexcept
{
    return static_cast<core::u32>(_impl->cells.size());
}

} // namespace lpl::ecs
