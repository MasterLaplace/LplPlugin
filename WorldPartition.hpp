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
    }

    [[nodiscard]] Partition *getChunk(const Vec3 &position) const
    {
        return _partitions.get(getChunkKey(position));
    }

    [[nodiscard]] Partition *addPartition(const Vec3 &position)
    {
        return getOrCreateChunk(position);
    }

    void step(float deltatime)
    {
        _transitQueue.clear();
        _partitions.forEach([&](Partition &partition){
            partition.physicsTick(deltatime, _transitQueue);
        });
        for (const auto &entity : _transitQueue)
        {
            if (Partition *partition = getOrCreateChunk(entity.position))
                partition->addEntity(entity);
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

    Partition *getOrCreateChunk(const Vec3 &position)
    {
        uint64_t key = getChunkKey(position);
        Partition *partition = _partitions.get(key);

        if (partition)
            return partition;

        float gridX = std::floor(position.x / _chunkSize) * _chunkSize;
        float gridZ = std::floor(position.z / _chunkSize) * _chunkSize;

        return _partitions.insert(key, Partition{{gridX, 0.f, gridZ}, _chunkSize});
    }

private:
    static constexpr uint64_t WORLD_CAPACITY = 1ULL << 16ul;
    FlatAtomicsHashMap<Partition> _partitions;
    std::vector<Partition::EntitySnapshot> _transitQueue;
    float _chunkSize;
};
