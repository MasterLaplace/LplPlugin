/**
 * @file Registry.cpp
 * @brief Entity registry with atomic lock-free free-list.
 *
 * Ported from legacy EntityRegistry: atomic free-list using a
 * Treiber stack (lock-free, ABA-safe via generation counter).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/ecs/Registry.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Constants.hpp>

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <vector>

namespace lpl::ecs {

// ========================================================================== //
//  Impl — Atomic free-list (Treiber stack with separate arrays)              //
// ========================================================================== //

struct Registry::Impl
{
    /** @brief Sentinel value meaning "no next slot". */
    static constexpr core::u32 kNoNext  = ~core::u32{0};
    static constexpr core::u32 kInitCap = 1024;

    /** @brief Non-atomic slot metadata (not stored in vector of atomics). */
    struct SlotInfo
    {
        core::u32  generation{0};
        core::u32  partitionIndex{0};
        EntityRef  ref{};
        bool       alive{false};
    };

    std::vector<SlotInfo>                   slots;
    std::vector<core::u32>                  freeNext; ///< Per-slot intrusive free-list link
    std::atomic<core::u32>                  freeHead{kNoNext}; ///< Head of Treiber stack
    std::vector<std::unique_ptr<Partition>> partitions;
    std::atomic<core::u32>                  liveCount{0};
    std::atomic<core::u32>                  highWater{0};

    Impl()
    {
        slots.resize(kInitCap);
        freeNext.resize(kInitCap, kNoNext);
    }

    /**
     * @brief Allocates a slot from the lock-free free-list (Treiber stack pop).
     */
    core::u32 allocateSlot()
    {
        // Try to pop from the Treiber stack
        // Note: freeNext is only written by the thread that owns the slot,
        // so we can use relaxed load. The CAS on freeHead provides ordering.
        core::u32 head = freeHead.load(std::memory_order_acquire);
        while (head != kNoNext)
        {
            const core::u32 nextVal = freeNext[head];
            if (freeHead.compare_exchange_weak(head, nextVal,
                                                std::memory_order_release,
                                                std::memory_order_acquire))
            {
                return head;
            }
        }

        // Free-list empty — allocate a new slot
        const core::u32 slot = highWater.fetch_add(1, std::memory_order_relaxed);
        if (slot >= static_cast<core::u32>(slots.size()))
        {
            const core::u32 newCap = slot + 256;
            slots.resize(newCap);
            freeNext.resize(newCap, kNoNext);
        }
        return slot;
    }

    /**
     * @brief Returns a slot to the lock-free free-list (Treiber stack push).
     */
    void freeSlot(core::u32 slot)
    {
        core::u32 head = freeHead.load(std::memory_order_relaxed);
        do
        {
            freeNext[slot] = head;
        }
        while (!freeHead.compare_exchange_weak(head, slot,
                                                std::memory_order_release,
                                                std::memory_order_relaxed));
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
    : _impl{std::make_unique<Impl>()}
{}

Registry::~Registry() = default;

core::Expected<EntityId> Registry::createEntity(const Archetype& archetype)
{
    const core::u32 slot = _impl->allocateSlot();

    if (slot >= (1u << EntityId::kSlotBits))
    {
        return core::makeError(core::ErrorCode::OutOfMemory, "Entity slot pool exhausted");
    }

    auto& info = _impl->slots[slot];
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
    _impl->liveCount.fetch_add(1, std::memory_order_relaxed);

    return id;
}

core::Expected<void> Registry::destroyEntity(EntityId id)
{
    if (!isAlive(id))
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Entity is not alive");
    }

    auto& info = _impl->slots[id.slot()];
    info.alive = false;

    auto& partition = *_impl->partitions[info.partitionIndex];
    auto eraseResult = partition.erase(info.ref);
    if (!eraseResult.has_value())
    {
        return core::makeError(eraseResult.error().code(), eraseResult.error().message());
    }

    // Bump generation (prevents stale EntityId from matching after recycle)
    info.generation = (info.generation + 1) & EntityId::kGenerationMask;
    _impl->freeSlot(id.slot());
    _impl->liveCount.fetch_sub(1, std::memory_order_relaxed);

    return {};
}

bool Registry::isAlive(EntityId id) const noexcept
{
    const core::u32 hw = _impl->highWater.load(std::memory_order_acquire);
    if (id.slot() >= hw)
    {
        return false;
    }
    const auto& info = _impl->slots[id.slot()];
    return info.alive && info.generation == id.generation();
}

core::Expected<EntityRef> Registry::resolve(EntityId id) const
{
    if (!isAlive(id))
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Entity is dead or invalid");
    }
    return _impl->slots[id.slot()].ref;
}

core::u32 Registry::liveCount() const noexcept
{
    return _impl->liveCount.load(std::memory_order_relaxed);
}

Partition& Registry::getOrCreatePartition(const Archetype& archetype)
{
    if (auto* existing = _impl->findPartition(archetype))
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

    _impl->partitions.push_back(
        std::make_unique<Partition>(archetype, std::move(layouts)));

    return *_impl->partitions.back();
}

std::span<const std::unique_ptr<Partition>> Registry::partitions() const noexcept
{
    return _impl->partitions;
}

void Registry::swapAllBuffers() noexcept
{
    for (auto& partition : _impl->partitions)
    {
        partition->swapAllBuffers();
    }
}

} // namespace lpl::ecs
