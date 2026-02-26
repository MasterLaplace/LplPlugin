// /////////////////////////////////////////////////////////////////////////////
/// @file Registry.cpp
/// @brief Entity registry implementation with generation tracking.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/ecs/Registry.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Constants.hpp>

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <vector>

namespace lpl::ecs {

// ========================================================================== //
//  Impl                                                                      //
// ========================================================================== //

struct Registry::Impl
{
    struct SlotInfo
    {
        core::u32  generation{0};
        core::u32  partitionIndex{0};
        EntityRef  ref{};
        bool       alive{false};
    };

    std::vector<SlotInfo>                   slots;
    std::queue<core::u32>                   freeSlots;
    std::vector<std::unique_ptr<Partition>> partitions;
    core::u32                               liveCount{0};

    core::u32 allocateSlot()
    {
        if (!freeSlots.empty())
        {
            core::u32 slot = freeSlots.front();
            freeSlots.pop();
            return slot;
        }

        const core::u32 slot = static_cast<core::u32>(slots.size());
        slots.emplace_back();
        return slot;
    }

    Partition* findPartition(const Archetype& arch)
    {
        for (auto& p : partitions)
        {
            if (p->archetype() == arch)
            {
                return p.get();
            }
        }
        return nullptr;
    }
};

// ========================================================================== //
//  Registry                                                                  //
// ========================================================================== //

Registry::Registry()
    : impl_{std::make_unique<Impl>()}
{}

Registry::~Registry() = default;

core::Expected<EntityId> Registry::createEntity(const Archetype& archetype)
{
    const core::u32 slot = impl_->allocateSlot();

    if (slot >= (1u << EntityId::kSlotBits))
    {
        return core::makeError(core::ErrorCode::OutOfMemory, "Entity slot pool exhausted");
    }

    auto& info = impl_->slots[slot];
    info.alive = true;

    const EntityId id{info.generation, slot};

    Partition& partition = getOrCreatePartition(archetype);
    auto refResult = partition.insert(id);
    if (!refResult.has_value())
    {
        info.alive = false;
        return core::makeError(refResult.error().code(), refResult.error().message());
    }

    info.ref = refResult.value();
    ++impl_->liveCount;

    return id;
}

core::Expected<void> Registry::destroyEntity(EntityId id)
{
    if (!isAlive(id))
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Entity is not alive");
    }

    auto& info = impl_->slots[id.slot()];
    info.alive = false;

    auto& partition = *impl_->partitions[info.partitionIndex];
    auto eraseResult = partition.erase(info.ref);
    if (!eraseResult.has_value())
    {
        return core::makeError(eraseResult.error().code(), eraseResult.error().message());
    }

    info.generation = (info.generation + 1) & EntityId::kGenerationMask;
    impl_->freeSlots.push(id.slot());
    --impl_->liveCount;

    return {};
}

bool Registry::isAlive(EntityId id) const noexcept
{
    if (id.slot() >= static_cast<core::u32>(impl_->slots.size()))
    {
        return false;
    }
    const auto& info = impl_->slots[id.slot()];
    return info.alive && info.generation == id.generation();
}

core::Expected<EntityRef> Registry::resolve(EntityId id) const
{
    if (!isAlive(id))
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Entity is dead or invalid");
    }
    return impl_->slots[id.slot()].ref;
}

core::u32 Registry::liveCount() const noexcept
{
    return impl_->liveCount;
}

Partition& Registry::getOrCreatePartition(const Archetype& archetype)
{
    if (auto* existing = impl_->findPartition(archetype))
    {
        return *existing;
    }

    std::vector<ComponentLayout> layouts;
    for (core::u32 i = 0; i < static_cast<core::u32>(ComponentId::Count); ++i)
    {
        const auto cid = static_cast<ComponentId>(i);
        if (archetype.has(cid))
        {
            layouts.push_back(ComponentLayout{cid, 0, 0});
        }
    }

    impl_->partitions.push_back(
        std::make_unique<Partition>(archetype, std::move(layouts)));

    return *impl_->partitions.back();
}

std::span<const std::unique_ptr<Partition>> Registry::partitions() const noexcept
{
    return impl_->partitions;
}

void Registry::swapAllBuffers() noexcept
{
    for (auto& partition : impl_->partitions)
    {
        partition->swapAllBuffers();
    }
}

} // namespace lpl::ecs
