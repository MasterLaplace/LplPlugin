#pragma once

#include "FlatAtomicsHashMap.hpp"
#include "Partition.hpp"
#include "Morton.hpp"
#include <cmath>

class WorldPartition {
public:
    WorldPartition() noexcept : _partitions(WORLD_CAPACITY), _chunkSize(255.f)
    {
        _transitQueue.reserve(1024u);
        _entityToChunk.assign(1000000u, INVALID_CHUNK_KEY);
    }

    [[nodiscard]] Partition *getChunk(const Vec3 &position) const
    {
        return _partitions.get(getChunkKey(position));
    }

    [[nodiscard]] Partition *getChunk(const uint64_t chunkKey) const
    {
        return _partitions.get(chunkKey);
    }

    void addEntity(const Partition::EntitySnapshot &entity)
    {
        uint64_t key = getChunkKey(entity.position);
        Partition *partition = getOrCreateChunk(entity.position, key);
        if (!partition)
            return;

        partition->addEntity(entity);
        updateEntityChunk(entity.id, key);
    }

    void updateEntityChunk(const uint32_t entityId, const uint64_t newChunkKey) noexcept
    {
        if (entityId < _entityToChunk.size())
            _entityToChunk[entityId] = newChunkKey;
    }

    [[nodiscard]] uint64_t getEntityChunkKey(const uint32_t entityId) const noexcept
    {
        return (entityId < _entityToChunk.size()) ? _entityToChunk[entityId] : INVALID_CHUNK_KEY;
    }

    void step(float deltatime)
    {
        _transitQueue.clear();
        _partitions.forEach([&](Partition &partition){
            partition.physicsTick(deltatime, _transitQueue);
        });
        for (const auto &entity : _transitQueue)
        {
            uint64_t key = getChunkKey(entity.position);
            if (Partition *partition = getOrCreateChunk(entity.position, key))
            {
                partition->addEntity(entity);
                updateEntityChunk(entity.id, key);
            }
        }
    }

private:
    [[nodiscard]] uint64_t getChunkKey(const Vec3 &position) const noexcept
    {
        auto x = static_cast<int>(std::floor(position.x / _chunkSize));
        auto z = static_cast<int>(std::floor(position.z / _chunkSize));
        const uint64_t bias = 1ULL << (20ul);
        uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(x) + static_cast<int64_t>(bias));
        uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>(z) + static_cast<int64_t>(bias));
        return Morton::encode2D(static_cast<uint32_t>(ux), static_cast<uint32_t>(uz));
    }

    Partition *getOrCreateChunk(const Vec3 &position, const uint64_t key)
    {
        Partition *partition = _partitions.get(key);

        if (partition)
            return partition;

        float gridX = std::floor(position.x / _chunkSize) * _chunkSize;
        float gridZ = std::floor(position.z / _chunkSize) * _chunkSize;

        return _partitions.insert(key, Partition{{gridX, 0.f, gridZ}, _chunkSize});
    }

private:
    static constexpr uint64_t WORLD_CAPACITY = 1ULL << 16ul;
    static constexpr uint64_t INVALID_CHUNK_KEY = std::numeric_limits<uint64_t>::max();
    FlatAtomicsHashMap<Partition> _partitions;
    std::vector<Partition::EntitySnapshot> _transitQueue;
    std::vector<uint64_t> _entityToChunk;
    float _chunkSize;
};
