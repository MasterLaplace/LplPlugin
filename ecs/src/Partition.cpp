/**
 * @file Partition.cpp
 * @brief Chunk and Partition implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/ecs/Partition.hpp>
#include <lpl/container/SparseSet.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/memory/PinnedAllocator.hpp>

#include <algorithm>
#include <cstring>
#include <array>
#include <optional>

namespace lpl::ecs {

// ========================================================================== //
//  Chunk::Impl                                                               //
// ========================================================================== //

struct Chunk::Impl
{
    using Alloc = memory::PinnedAllocator<core::byte>;

    Archetype                                           archetype;
    std::vector<ComponentLayout>                        layouts;
    std::array<std::pair<void*, void*>, static_cast<core::usize>(ComponentId::Count)> buffers{};
    std::vector<EntityId>                               entities;
    /// Sparse set for O(1) slot → localIndex lookup.
    /// Capacity = 2^kSlotBits (16 384) to cover all valid entity slots.
    container::SparseSet<core::u32>                     slotToLocal{1u << EntityId::kSlotBits};
    core::u32                                           count{0};
    Alloc                                               alloc{};

    explicit Impl(const Archetype& arch,
                  std::span<const ComponentLayout> lyts)
        : archetype{arch}
        , layouts{lyts.begin(), lyts.end()}
    {
        entities.resize(kChunkCapacity);

        for (const auto& layout : layouts)
        {
            if (!archetype.has(layout.id))
            {
                continue;
            }

            const core::usize bytes = static_cast<core::usize>(layout.size) * kChunkCapacity;
            auto* front = alloc.allocate(bytes);
            auto* back  = alloc.allocate(bytes);
            std::memset(front, 0, bytes);
            std::memset(back,  0, bytes);
            buffers[static_cast<core::usize>(layout.id)] = {front, back};
        }
    }

    ~Impl()
    {
        for (auto& pair : buffers)
        {
            if (pair.first) alloc.deallocate(static_cast<core::byte*>(pair.first), 0);
            if (pair.second) alloc.deallocate(static_cast<core::byte*>(pair.second), 0);
        }
    }
};

// ========================================================================== //
//  Chunk                                                                     //
// ========================================================================== //

Chunk::Chunk(const Archetype& archetype,
             std::span<const ComponentLayout> layouts)
    : _impl{std::make_unique<Impl>(archetype, layouts)}
{}

Chunk::~Chunk() = default;

core::u32 Chunk::count() const noexcept
{
    return _impl->count;
}

bool Chunk::isFull() const noexcept
{
    return _impl->count >= kChunkCapacity;
}

core::Expected<core::u32> Chunk::add(EntityId id)
{
    if (isFull())
    {
        return core::makeError(core::ErrorCode::OutOfMemory, "Chunk is full");
    }

    const core::u32 idx = _impl->count++;
    _impl->entities[idx] = id;
    _impl->slotToLocal.insert(id.slot(), idx);
    return idx;
}

core::Expected<EntityId> Chunk::remove(core::u32 localIndex)
{
    if (localIndex >= _impl->count)
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Invalid local index");
    }

    const core::u32 last = _impl->count - 1;
    EntityId        removed  = _impl->entities[localIndex];   // entity being evicted
    EntityId        swapped  = _impl->entities[last];         // entity moved into the gap

    // Remove the evicted entity from the sparse set first.
    _impl->slotToLocal.remove(removed.slot());

    if (localIndex != last)
    {
        _impl->entities[localIndex] = _impl->entities[last];
        // Update the moved entity's mapping to its new local index.
        _impl->slotToLocal.insert(swapped.slot(), localIndex);

        for (const auto& layout : _impl->layouts)
        {
            if (!_impl->archetype.has(layout.id))
            {
                continue;
            }

            const core::usize cid = static_cast<core::usize>(layout.id);
            auto* front = static_cast<core::byte*>(_impl->buffers[cid].first);
            auto* back  = static_cast<core::byte*>(_impl->buffers[cid].second);
            const core::u32 sz = layout.size;

            std::memcpy(front + static_cast<core::usize>(localIndex) * sz,
                        front + static_cast<core::usize>(last) * sz,
                        sz);
            std::memcpy(back  + static_cast<core::usize>(localIndex) * sz,
                        back  + static_cast<core::usize>(last) * sz,
                        sz);
        }
    }

    --_impl->count;
    return swapped;
}

std::optional<core::u32> Chunk::findLocalIndex(EntityId id) const noexcept
{
    const core::u32* p = _impl->slotToLocal.find(id.slot());
    if (!p)
        return std::nullopt;
    // Guard: confirm the entity at that index still carries the same generation
    // (handles the case where the slot was recycled but the chunk still lives).
    const core::u32 local = *p;
    if (local >= _impl->count || _impl->entities[local] != id)
        return std::nullopt;
    return local;
}

const void* Chunk::readComponent(ComponentId id) const noexcept
{
    const core::usize cid = static_cast<core::usize>(id);
    if (cid >= static_cast<core::usize>(ComponentId::Count))
    {
        return nullptr;
    }
    return _impl->buffers[cid].first;
}

void* Chunk::writeComponent(ComponentId id) noexcept
{
    const core::usize cid = static_cast<core::usize>(id);
    if (cid >= static_cast<core::usize>(ComponentId::Count))
    {
        return nullptr;
    }
    return _impl->buffers[cid].second;
}

void Chunk::swapBuffers() noexcept
{
    // Copy back (write) → front (read) so the new write buffer starts
    // with up-to-date data after the swap (prevents 2-frame-stale reads).
    for (const auto& layout : _impl->layouts)
    {
        if (!_impl->archetype.has(layout.id))
            continue;

        const core::usize cid = static_cast<core::usize>(layout.id);
        const core::usize bytes = static_cast<core::usize>(layout.size) * static_cast<core::usize>(_impl->count);
        std::memcpy(_impl->buffers[cid].first, _impl->buffers[cid].second, bytes);

        std::swap(_impl->buffers[cid].first, _impl->buffers[cid].second);
    }
}

const Archetype& Chunk::archetype() const noexcept
{
    return _impl->archetype;
}

std::span<const EntityId> Chunk::entities() const noexcept
{
    return {_impl->entities.data(), _impl->count};
}


// ========================================================================== //

Partition::Partition(Archetype archetype,
                     std::vector<ComponentLayout> layouts)
    : _archetype{std::move(archetype)}
    , _layouts{std::move(layouts)}
{}

Partition::~Partition() = default;

core::Expected<EntityRef> Partition::insert(EntityId id)
{
    for (core::u32 ci = 0; ci < static_cast<core::u32>(_chunks.size()); ++ci)
    {
        if (!_chunks[ci]->isFull())
        {
            auto result = _chunks[ci]->add(id);
            if (!result.has_value())
            {
                return core::makeError(result.error().code(), result.error().message());
            }
            return EntityRef{id, ci, result.value()};
        }
    }

    _chunks.push_back(std::make_unique<Chunk>(_archetype, _layouts));
    const core::u32 ci = static_cast<core::u32>(_chunks.size()) - 1;

    auto result = _chunks[ci]->add(id);
    if (!result.has_value())
    {
        return core::makeError(result.error().code(), result.error().message());
    }
    return EntityRef{id, ci, result.value()};
}

core::Expected<EntityId> Partition::erase(const EntityRef& ref)
{
    if (ref.chunkIndex >= static_cast<core::u32>(_chunks.size()))
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Invalid chunk index");
    }

    auto result = _chunks[ref.chunkIndex]->remove(ref.localIndex);
    if (!result.has_value())
    {
        return core::makeError(result.error().code(), result.error().message());
    }
    return result.value();
}

core::u32 Partition::entityCount() const noexcept
{
    core::u32 total = 0;
    for (const auto& chunk : _chunks)
    {
        total += chunk->count();
    }
    return total;
}

std::span<const std::unique_ptr<Chunk>> Partition::chunks() const noexcept
{
    return _chunks;
}

void Partition::swapAllBuffers() noexcept
{
    for (auto& chunk : _chunks)
    {
        chunk->swapBuffers();
    }
}

const Archetype& Partition::archetype() const noexcept
{
    return _archetype;
}

} // namespace lpl::ecs
