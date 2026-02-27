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
#include <lpl/core/Assert.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace lpl::ecs {

// ========================================================================== //
//  Chunk::Impl                                                               //
// ========================================================================== //

struct Chunk::Impl
{
    Archetype                                           archetype;
    std::vector<ComponentLayout>                        layouts;
    std::unordered_map<ComponentId, std::pair<void*, void*>> buffers;
    std::vector<EntityId>                               entities;
    core::u32                                           count{0};

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
            void* front = std::aligned_alloc(layout.alignment, bytes);
            void* back  = std::aligned_alloc(layout.alignment, bytes);
            std::memset(front, 0, bytes);
            std::memset(back,  0, bytes);
            buffers[layout.id] = {front, back};
        }
    }

    ~Impl()
    {
        for (auto& [id, pair] : buffers)
        {
            std::free(pair.first);
            std::free(pair.second);
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
    return idx;
}

core::Expected<EntityId> Chunk::remove(core::u32 localIndex)
{
    if (localIndex >= _impl->count)
    {
        return core::makeError(core::ErrorCode::InvalidArgument, "Invalid local index");
    }

    const core::u32 last = _impl->count - 1;
    EntityId swapped = _impl->entities[last];

    if (localIndex != last)
    {
        _impl->entities[localIndex] = _impl->entities[last];

        for (auto& [compId, pair] : _impl->buffers)
        {
            const auto it = std::find_if(
                _impl->layouts.begin(), _impl->layouts.end(),
                [compId = compId](const ComponentLayout& l) { return l.id == compId; });

            if (it == _impl->layouts.end())
            {
                continue;
            }

            auto* front = static_cast<core::byte*>(pair.first);
            auto* back  = static_cast<core::byte*>(pair.second);
            const core::u32 sz = it->size;

            std::memcpy(front + localIndex * sz, front + last * sz, sz);
            std::memcpy(back  + localIndex * sz, back  + last * sz, sz);
        }
    }

    --_impl->count;
    return swapped;
}

const void* Chunk::readComponent(ComponentId id) const noexcept
{
    auto it = _impl->buffers.find(id);
    if (it == _impl->buffers.end())
    {
        return nullptr;
    }
    return it->second.first;
}

void* Chunk::writeComponent(ComponentId id) noexcept
{
    auto it = _impl->buffers.find(id);
    if (it == _impl->buffers.end())
    {
        return nullptr;
    }
    return it->second.second;
}

void Chunk::swapBuffers() noexcept
{
    for (auto& [id, pair] : _impl->buffers)
    {
        std::swap(pair.first, pair.second);
    }
}

const Archetype& Chunk::archetype() const noexcept
{
    return _impl->archetype;
}

// ========================================================================== //
//  Partition                                                                  //
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

core::Expected<void> Partition::erase(const EntityRef& ref)
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
    return {};
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
