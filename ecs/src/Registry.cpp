#include <lpl/core/Assert.hpp>
#include <lpl/core/Constants.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/std/vector.hpp>

#include <algorithm>
#include <atomic>

namespace lpl::ecs {

struct Registry::Impl {
    memory::IAllocator *allocator = nullptr; /**< Optional arena backing chunk storage (see Registry::setAllocator). */

    ComponentBytes swappedBytes{}; /**< Bytes the swaps have copied, per component, since construction. */

    /** @brief Sentinel value meaning "no next slot". */
    static constexpr core::u32 kNoNext = ~core::u32{0};
    static constexpr core::u32 kInitCap = 1024;

    /** @brief Non-atomic slot metadata (not stored in vector of atomics). */
    struct SlotInfo {
        core::u32 generation{0};
        core::u32 partitionIndex{0}; /**< Which partition holds the entity, so destroyEntity finds the correct one. */
        EntityRef ref{};
        bool alive{false};
    };

    lpl::pmr::vector<SlotInfo> slots;
    lpl::pmr::vector<core::u32> freeNext;     /**< Per-slot intrusive free-list link */
    std::atomic<core::u32> freeHead{kNoNext}; /**< Head of Treiber stack */
    lpl::pmr::vector<lpl::pmr::unique_ptr<Partition>> partitions;
    std::atomic<core::u32> liveCount{0};
    std::atomic<core::u32> highWater{0};

    Impl()
    {
        slots.resize(kInitCap);
        freeNext.resize(kInitCap, kNoNext);
    }

    /**
     * @brief Allocates a slot from the lock-free free-list (Treiber stack pop).
     *
     * freeNext is only written by the thread that owns the slot; the CAS on freeHead provides
     * ordering. A slot on the free-list may since have been reserved by createEntityWithId (which
     * does not unlink it): it is dropped and popping goes on, so a network-adopted id is never handed
     * out locally. When the free-list is empty, a new slot is allocated.
     */
    core::u32 allocateSlot()
    {
        core::u32 head = freeHead.load(std::memory_order_acquire);
        while (head != kNoNext)
        {
            const core::u32 nextVal = freeNext[head];
            if (freeHead.compare_exchange_weak(head, nextVal, std::memory_order_release, std::memory_order_acquire))
            {
                if (slots[head].alive)
                {
                    head = freeHead.load(std::memory_order_acquire);
                    continue;
                }
                return head;
            }
        }

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
        } while (!freeHead.compare_exchange_weak(head, slot, std::memory_order_release, std::memory_order_relaxed));
    }

    Partition *findPartition(const Archetype &arch)
    {
        for (auto &p : partitions)
        {
            if (p->archetype() == arch)
            {
                return p.get();
            }
        }
        return nullptr;
    }
};

Registry::Registry() : _impl{lpl::pmr::make_unique<Impl>()} {}

void Registry::setAllocator(memory::IAllocator *allocator) noexcept { _impl->allocator = allocator; }

Registry::~Registry() = default;

core::Expected<EntityId> Registry::createEntity(const Archetype &archetype)
{
    const core::u32 slot = _impl->allocateSlot();

    if (slot >= (1u << EntityId::kSlotBits))
    {
        return core::makeError(core::ErrorCode::OutOfMemory, "Entity slot pool exhausted");
    }

    auto &info = _impl->slots[slot];
    info.alive = true;

    const EntityId id{info.generation, slot};

    Partition &partition = getOrCreatePartition(archetype);
    auto refResult = partition.insert(id);
    if (!refResult.has_value())
    {
        info.alive = false;
        return core::makeError(refResult.error().code(), refResult.error().message());
    }

    info.ref = refResult.value();

    for (core::u32 pi = 0; pi < static_cast<core::u32>(_impl->partitions.size()); ++pi)
    {
        if (_impl->partitions[pi].get() == &partition)
        {
            info.partitionIndex = pi;
            break;
        }
    }

    _impl->liveCount.fetch_add(1, std::memory_order_relaxed);

    return id;
}

core::Expected<EntityId> Registry::createEntityWithId(EntityId id, const Archetype &archetype)
{
    const core::u32 slot = id.slot();
    if (slot >= (1u << EntityId::kSlotBits))
    {
        return core::makeError(core::ErrorCode::OutOfMemory, "Entity slot pool exhausted");
    }

    if (slot >= static_cast<core::u32>(_impl->slots.size()))
    {
        const core::u32 newCap = slot + 256;
        _impl->slots.resize(newCap);
        _impl->freeNext.resize(newCap, Impl::kNoNext);
    }
    core::u32 hw = _impl->highWater.load(std::memory_order_acquire);
    while (slot >= hw)
    {
        if (_impl->highWater.compare_exchange_weak(hw, slot + 1, std::memory_order_release, std::memory_order_acquire))
        {
            for (core::u32 s = hw; s < slot; ++s)
                _impl->freeSlot(s);
            break;
        }
    }

    auto &info = _impl->slots[slot];
    if (info.alive)
    {
        const EntityId existing{info.generation, slot};
        if (existing == id)
            return id;
        return core::makeError(core::ErrorCode::AlreadyExists, "Slot already holds a different live entity");
    }

    info.generation = id.generation();
    info.alive = true;

    Partition &partition = getOrCreatePartition(archetype);
    auto refResult = partition.insert(id);
    if (!refResult.has_value())
    {
        info.alive = false;
        return core::makeError(refResult.error().code(), refResult.error().message());
    }

    info.ref = refResult.value();

    for (core::u32 pi = 0; pi < static_cast<core::u32>(_impl->partitions.size()); ++pi)
    {
        if (_impl->partitions[pi].get() == &partition)
        {
            info.partitionIndex = pi;
            break;
        }
    }

    _impl->liveCount.fetch_add(1, std::memory_order_relaxed);

    return id;
}

core::Expected<void> Registry::destroyEntity(EntityId id)
{
    if (!isAlive(id))
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Entity is not alive");
    }

    auto &info = _impl->slots[id.slot()];
    info.alive = false;

    auto &partition = *_impl->partitions[info.partitionIndex];
    auto eraseResult = partition.erase(info.ref);
    if (!eraseResult.has_value())
    {
        return core::makeError(eraseResult.error().code(), eraseResult.error().message());
    }

    EntityId swappedId = eraseResult.value();
    if (swappedId != id)
    {
        _impl->slots[swappedId.slot()].ref.localIndex = info.ref.localIndex;
    }

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
    const auto &info = _impl->slots[id.slot()];
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

Chunk *Registry::chunkOf(EntityId id, core::u32 &outRow) const noexcept
{
    const auto ref = resolve(id);
    if (!ref)
        return nullptr;
    const core::u32 chunkIndex = ref.value().chunkIndex;
    const core::u32 row = ref.value().localIndex;

    for (const auto &partition : partitions())
    {
        if (!partition)
            continue;
        const auto &chunks = partition->chunks();
        if (chunkIndex >= chunks.size() || !chunks[chunkIndex])
            continue;
        Chunk *chunk = chunks[chunkIndex].get();
        if (row >= chunk->count())
            continue;
        const std::span<const EntityId> entities = chunk->entities();
        if (row < entities.size() && entities[row] == id)
        {
            outRow = row;
            return chunk;
        }
    }
    return nullptr;
}

core::u32 Registry::liveCount() const noexcept { return _impl->liveCount.load(std::memory_order_relaxed); }

Partition &Registry::getOrCreatePartition(const Archetype &archetype)
{
    if (auto *existing = _impl->findPartition(archetype))
    {
        return *existing;
    }

    lpl::pmr::vector<ComponentLayout> layouts;
    for (core::u32 i = 0; i < static_cast<core::u32>(ComponentId::Count); ++i)
    {
        const auto cid = static_cast<ComponentId>(i);
        if (archetype.has(cid))
        {
            layouts.push_back(defaultLayout(cid));
        }
    }

    _impl->partitions.push_back(lpl::pmr::make_unique<Partition>(archetype, std::move(layouts), _impl->allocator));

    return *_impl->partitions.back();
}

std::span<const lpl::pmr::unique_ptr<Partition>> Registry::partitions() const noexcept { return _impl->partitions; }

void Registry::swapAllBuffers() noexcept
{
    for (auto &partition : _impl->partitions)
    {
        partition->swapAllBuffers(_impl->swappedBytes);
    }
}

core::u64 Registry::swappedBytes() const noexcept
{
    core::u64 total = 0u;
    for (const core::u64 bytes : _impl->swappedBytes)
        total += bytes;
    return total;
}

core::u64 Registry::swappedBytes(ComponentId id) const noexcept
{
    const core::usize index = static_cast<core::usize>(id);
    return index < _impl->swappedBytes.size() ? _impl->swappedBytes[index] : 0u;
}

} // namespace lpl::ecs
