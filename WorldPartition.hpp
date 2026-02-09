#pragma once

#include "FlatAtomicsHashMap.hpp"
#include "Partition.hpp"
#include "Morton.hpp"

class WorldPartition {
public:
    WorldPartition() noexcept : _partitions(WORLD_CAPACITY), _chunkSize(255.f) {}

    [[nodiscard]] Partition *getChunk(const Vec3 &position) const
    {
        return _partitions.get(getChunkKey(position));
    }

private:
    [[nodiscard]] uint64_t getChunkKey(const Vec3 &position) const noexcept
    {
        auto x = static_cast<int>(std::floor(position.x / _chunkSize));
        auto z = static_cast<int>(std::floor(position.z / _chunkSize));
        const uint64_t bias = 1ul << (20ul);
        uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(x) + static_cast<int64_t>(bias));
        uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>(z) + static_cast<int64_t>(bias));
        return Morton::encode2D(static_cast<uint32_t>(ux), static_cast<uint32_t>(uz));
    }

private:
    static constexpr uint64_t WORLD_CAPACITY = 1ul << 16ul;
    FlatAtomicsHashMap<Partition> _partitions;
    float _chunkSize;
};
