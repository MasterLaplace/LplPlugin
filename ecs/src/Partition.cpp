#include <lpl/container/SparseSet.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/memory/PinnedAllocator.hpp>
#include <lpl/std/cstring.hpp>

#include <algorithm>
#include <array>
#include <optional>

namespace lpl::ecs {

static_assert(static_cast<core::usize>(ComponentId::Count) <= 32u, "a chunk marks its written components in one u32");

struct Chunk::Impl {
    using Alloc = memory::PinnedAllocator<core::byte>;

    Archetype archetype;
    lpl::pmr::vector<ComponentLayout> layouts;
    std::array<std::pair<void *, void *>, static_cast<core::usize>(ComponentId::Count)> buffers{};
    lpl::pmr::vector<EntityId> entities;
    container::SparseSet<core::u32> slotToLocal{1u << EntityId::kSlotBits}; /**< O(1) slot → localIndex, all slots. */
    core::u32 count{0};
    core::u32 written{0}; /**< One bit per ComponentId: handed out for writing since the last swap. */
    Alloc alloc{};

    Impl(const Archetype &arch, std::span<const ComponentLayout> lyts, memory::IAllocator *external)
        : archetype{arch}, layouts{lyts.begin(), lyts.end()}, externalAllocator{external}
    {
        entities.resize(kChunkCapacity);

        for (const auto &layout : layouts)
        {
            if (!archetype.has(layout.id))
            {
                continue;
            }

            const core::usize bytes = static_cast<core::usize>(layout.size) * kChunkCapacity;
            auto *front = allocateBytes(bytes);
            auto *back = allocateBytes(bytes);
            if (front == nullptr || back == nullptr)
            {
                releaseBytes(front);
                releaseBytes(back);
                continue;
            }
            lpl::pmr::memset(front, 0, bytes);
            lpl::pmr::memset(back, 0, bytes);
            buffers[static_cast<core::usize>(layout.id)] = {front, back};
        }
    }

    ~Impl()
    {
        for (auto &pair : buffers)
        {
            releaseBytes(pair.first);
            releaseBytes(pair.second);
        }
    }

    /**
     * @brief Chunk storage comes from the World's arena when one was injected, and
     *        from the heap otherwise.
     *
     * Same bytes either way — only the source differs.
     */
    [[nodiscard]] void *allocateBytes(core::usize bytes)
    {
        return externalAllocator ? externalAllocator->allocate(bytes) : alloc.allocate(bytes);
    }

    /**
     * @brief Releases a block @ref allocateBytes returned, unless an arena holds it: an arena
     *        reclaims in bulk on reset, never per block.
     */
    void releaseBytes(void *pointer)
    {
        if (pointer == nullptr)
            return;
        if (externalAllocator == nullptr)
            alloc.deallocate(static_cast<core::byte *>(pointer), 0);
    }

    memory::IAllocator *externalAllocator = nullptr;
};

Chunk::Chunk(const Archetype &archetype, std::span<const ComponentLayout> layouts, memory::IAllocator *allocator)
    : _impl{lpl::pmr::make_unique<Impl>(archetype, layouts, allocator)}
{
}

Chunk::~Chunk() = default;

core::u32 Chunk::count() const noexcept { return _impl->count; }

bool Chunk::isFull() const noexcept { return _impl->count >= kChunkCapacity; }

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
    EntityId removed = _impl->entities[localIndex];
    EntityId swapped = _impl->entities[last];

    _impl->slotToLocal.remove(removed.slot());

    if (localIndex != last)
    {
        _impl->entities[localIndex] = _impl->entities[last];
        _impl->slotToLocal.insert(swapped.slot(), localIndex);

        for (const auto &layout : _impl->layouts)
        {
            if (!_impl->archetype.has(layout.id))
            {
                continue;
            }

            const core::usize cid = static_cast<core::usize>(layout.id);
            auto *front = static_cast<core::byte *>(_impl->buffers[cid].first);
            auto *back = static_cast<core::byte *>(_impl->buffers[cid].second);
            const core::u32 sz = layout.size;

            lpl::pmr::memcpy(front + static_cast<core::usize>(localIndex) * sz,
                             front + static_cast<core::usize>(last) * sz, sz);
            lpl::pmr::memcpy(back + static_cast<core::usize>(localIndex) * sz,
                             back + static_cast<core::usize>(last) * sz, sz);
        }
    }

    --_impl->count;
    return swapped;
}

std::optional<core::u32> Chunk::findLocalIndex(EntityId id) const noexcept
{
    const core::u32 *p = _impl->slotToLocal.find(id.slot());
    if (!p)
        return std::nullopt;
    const core::u32 local = *p;
    if (local >= _impl->count || _impl->entities[local] != id)
        return std::nullopt;
    return local;
}

const void *Chunk::readComponent(ComponentId id) const noexcept
{
    const core::usize cid = static_cast<core::usize>(id);
    if (cid >= static_cast<core::usize>(ComponentId::Count))
    {
        return nullptr;
    }
    return _impl->buffers[cid].first;
}

void *Chunk::writeComponent(ComponentId id) noexcept
{
    const core::usize cid = static_cast<core::usize>(id);
    if (cid >= static_cast<core::usize>(ComponentId::Count))
    {
        return nullptr;
    }
    _impl->written |= 1u << cid;
    return _impl->buffers[cid].second;
}

const void *Chunk::latestComponent(ComponentId id) const noexcept
{
    const core::usize cid = static_cast<core::usize>(id);
    if (cid >= static_cast<core::usize>(ComponentId::Count))
    {
        return nullptr;
    }
    return _impl->buffers[cid].second;
}

void Chunk::swapBuffers(ComponentBytes &outBytes) noexcept
{
    for (const auto &layout : _impl->layouts)
    {
        const core::usize cid = static_cast<core::usize>(layout.id);
        if (!_impl->archetype.has(layout.id) || (_impl->written & (1u << cid)) == 0u)
            continue;

        const core::usize bytes = static_cast<core::usize>(layout.size) * static_cast<core::usize>(_impl->count);
        lpl::pmr::memcpy(_impl->buffers[cid].first, _impl->buffers[cid].second, bytes);
        outBytes[cid] += bytes;

        std::swap(_impl->buffers[cid].first, _impl->buffers[cid].second);
    }
    _impl->written = 0u;
}

const Archetype &Chunk::archetype() const noexcept { return _impl->archetype; }

std::span<const EntityId> Chunk::entities() const noexcept { return {_impl->entities.data(), _impl->count}; }

Partition::Partition(Archetype archetype, lpl::pmr::vector<ComponentLayout> layouts, memory::IAllocator *allocator)
    : _archetype{std::move(archetype)}, _layouts{std::move(layouts)}, _allocator{allocator}
{
}

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

    _chunks.push_back(lpl::pmr::make_unique<Chunk>(_archetype, _layouts, _allocator));
    const core::u32 ci = static_cast<core::u32>(_chunks.size()) - 1;

    auto result = _chunks[ci]->add(id);
    if (!result.has_value())
    {
        return core::makeError(result.error().code(), result.error().message());
    }
    return EntityRef{id, ci, result.value()};
}

core::Expected<EntityId> Partition::erase(const EntityRef &ref)
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
    for (const auto &chunk : _chunks)
    {
        total += chunk->count();
    }
    return total;
}

std::span<const lpl::pmr::unique_ptr<Chunk>> Partition::chunks() const noexcept { return _chunks; }

void Partition::swapAllBuffers(ComponentBytes &outBytes) noexcept
{
    for (auto &chunk : _chunks)
    {
        chunk->swapBuffers(outBytes);
    }
}

const Archetype &Partition::archetype() const noexcept { return _archetype; }

} // namespace lpl::ecs
