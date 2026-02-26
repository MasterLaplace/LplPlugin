// /////////////////////////////////////////////////////////////////////////////
/// @file WorldPartition.cpp
/// @brief WorldPartition implementation.
// /////////////////////////////////////////////////////////////////////////////

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
    : impl_{std::make_unique<Impl>(cellSize)}
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

    auto it = impl_->entityToMorton.find(raw);
    if (it != impl_->entityToMorton.end())
    {
        if (it->second == morton)
        {
            return {};
        }

        impl_->cells[it->second].erase(raw);
        if (impl_->cells[it->second].empty())
        {
            impl_->cells.erase(it->second);
        }
    }

    impl_->cells[morton].insert(raw);
    impl_->entityToMorton[raw] = morton;

    return {};
}

core::Expected<void> WorldPartition::remove(EntityId id)
{
    const core::u32 raw = id.raw();
    auto it = impl_->entityToMorton.find(raw);
    if (it == impl_->entityToMorton.end())
    {
        return core::makeError(core::ErrorCode::NotFound, "Entity not in world partition");
    }

    impl_->cells[it->second].erase(raw);
    if (impl_->cells[it->second].empty())
    {
        impl_->cells.erase(it->second);
    }
    impl_->entityToMorton.erase(it);

    return {};
}

void WorldPartition::queryRadius(
    const math::Vec3<math::Fixed32>& /*center*/,
    math::Fixed32 /*radius*/,
    std::vector<EntityId>& /*results*/) const
{
    LPL_ASSERT(false && "queryRadius not yet implemented");
}

core::u64 WorldPartition::mortonForPosition(const math::Vec3<math::Fixed32>& pos) const noexcept
{
    const auto toGrid = [&](math::Fixed32 v) -> core::i32 {
        const auto intVal = (v / impl_->cellSize).toInt();
        return intVal + static_cast<core::i32>(core::kMortonBias);
    };

    return math::morton::encode3D(toGrid(pos.x), toGrid(pos.y), toGrid(pos.z));
}

} // namespace lpl::ecs
