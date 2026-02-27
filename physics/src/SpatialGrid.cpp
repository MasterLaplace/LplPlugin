/**
 * @file SpatialGrid.cpp
 * @brief Uniform spatial hash grid implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/physics/SpatialGrid.hpp>
#include <lpl/core/Assert.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lpl::physics {

struct SpatialGrid::Impl
{
    math::Fixed32                                         cellSize;
    std::unordered_map<core::u64, std::unordered_set<core::u32>> cells;
    std::unordered_map<core::u32, math::AABB<math::Fixed32>>     objects;

    explicit Impl(math::Fixed32 cs) : cellSize{cs} {}

    [[nodiscard]] core::i32 toCell(math::Fixed32 v) const
    {
        return (v / cellSize).toInt();
    }

    [[nodiscard]] core::u64 hashCell(core::i32 cx, core::i32 cy, core::i32 cz) const
    {
        auto h = static_cast<core::u64>(static_cast<core::u32>(cx));
        h ^= static_cast<core::u64>(static_cast<core::u32>(cy)) * 0x9E3779B97F4A7C15ULL;
        h ^= static_cast<core::u64>(static_cast<core::u32>(cz)) * 0x517CC1B727220A95ULL;
        return h;
    }

    void insertCells(core::u32 objectId, const math::AABB<math::Fixed32>& aabb)
    {
        const core::i32 minCx = toCell(aabb.min.x);
        const core::i32 minCy = toCell(aabb.min.y);
        const core::i32 minCz = toCell(aabb.min.z);
        const core::i32 maxCx = toCell(aabb.max.x);
        const core::i32 maxCy = toCell(aabb.max.y);
        const core::i32 maxCz = toCell(aabb.max.z);

        for (core::i32 cx = minCx; cx <= maxCx; ++cx)
        {
            for (core::i32 cy = minCy; cy <= maxCy; ++cy)
            {
                for (core::i32 cz = minCz; cz <= maxCz; ++cz)
                {
                    cells[hashCell(cx, cy, cz)].insert(objectId);
                }
            }
        }
    }

    void removeCells(core::u32 objectId, const math::AABB<math::Fixed32>& aabb)
    {
        const core::i32 minCx = toCell(aabb.min.x);
        const core::i32 minCy = toCell(aabb.min.y);
        const core::i32 minCz = toCell(aabb.min.z);
        const core::i32 maxCx = toCell(aabb.max.x);
        const core::i32 maxCy = toCell(aabb.max.y);
        const core::i32 maxCz = toCell(aabb.max.z);

        for (core::i32 cx = minCx; cx <= maxCx; ++cx)
        {
            for (core::i32 cy = minCy; cy <= maxCy; ++cy)
            {
                for (core::i32 cz = minCz; cz <= maxCz; ++cz)
                {
                    auto it = cells.find(hashCell(cx, cy, cz));
                    if (it != cells.end())
                    {
                        it->second.erase(objectId);
                        if (it->second.empty())
                        {
                            cells.erase(it);
                        }
                    }
                }
            }
        }
    }
};

SpatialGrid::SpatialGrid(math::Fixed32 cellSize)
    : _impl{std::make_unique<Impl>(cellSize)}
{
    LPL_ASSERT(cellSize > math::Fixed32{0});
}

SpatialGrid::~SpatialGrid() = default;

void SpatialGrid::insert(core::u32 objectId, const math::AABB<math::Fixed32>& aabb)
{
    _impl->objects[objectId] = aabb;
    _impl->insertCells(objectId, aabb);
}

void SpatialGrid::update(core::u32 objectId, const math::AABB<math::Fixed32>& aabb)
{
    auto it = _impl->objects.find(objectId);
    if (it != _impl->objects.end())
    {
        _impl->removeCells(objectId, it->second);
    }
    _impl->objects[objectId] = aabb;
    _impl->insertCells(objectId, aabb);
}

void SpatialGrid::remove(core::u32 objectId)
{
    auto it = _impl->objects.find(objectId);
    LPL_ASSERT(it != _impl->objects.end());
    _impl->removeCells(objectId, it->second);
    _impl->objects.erase(it);
}

void SpatialGrid::query(const math::AABB<math::Fixed32>& region,
                        const std::function<void(core::u32)>& callback) const
{
    std::unordered_set<core::u32> visited;

    const core::i32 minCx = _impl->toCell(region.min.x);
    const core::i32 minCy = _impl->toCell(region.min.y);
    const core::i32 minCz = _impl->toCell(region.min.z);
    const core::i32 maxCx = _impl->toCell(region.max.x);
    const core::i32 maxCy = _impl->toCell(region.max.y);
    const core::i32 maxCz = _impl->toCell(region.max.z);

    for (core::i32 cx = minCx; cx <= maxCx; ++cx)
    {
        for (core::i32 cy = minCy; cy <= maxCy; ++cy)
        {
            for (core::i32 cz = minCz; cz <= maxCz; ++cz)
            {
                auto it = _impl->cells.find(_impl->hashCell(cx, cy, cz));
                if (it == _impl->cells.end())
                {
                    continue;
                }

                for (core::u32 objId : it->second)
                {
                    if (visited.insert(objId).second)
                    {
                        const auto& objAabb = _impl->objects.at(objId);
                        if (objAabb.intersects(region))
                        {
                            callback(objId);
                        }
                    }
                }
            }
        }
    }
}

void SpatialGrid::rebuild()
{
    // no-op for hash grid
}

core::u32 SpatialGrid::count() const noexcept
{
    return static_cast<core::u32>(_impl->objects.size());
}

} // namespace lpl::physics
