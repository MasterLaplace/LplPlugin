#pragma once

#include "FlatAtomicsHashMap.hpp"
#include "Partition.hpp"
#include "EntityRegistry.hpp"
#include "Morton.hpp"
#include <cmath>

class WorldPartition {
public:
    WorldPartition() : _partitions(WORLD_CAPACITY), _chunkSize(255.f)
    {
        _transitQueue.reserve(1024u);
    }

    [[nodiscard]] Partition *getChunk(const Vec3 &position) const
    {
        return _partitions.get(getChunkKey(position));
    }

    [[nodiscard]] Partition *getChunk(const uint64_t chunkKey) const
    {
        return _partitions.get(chunkKey);
    }

    /**
     * @brief Ajoute une entité dans le monde.
     * @return Smart handle (generation | slot), ou UINT32_MAX en cas d'échec.
     */
    uint32_t addEntity(const Partition::EntitySnapshot &entity)
    {
        uint64_t key = getChunkKey(entity.position);
        Partition *partition = getOrCreateChunk(entity.position, key);
        if (!partition)
            return UINT32_MAX;

        partition->addEntity(entity);
        return _registry.registerEntity(entity.id, key);
    }

    /**
     * @brief Supprime une entité du monde par son ID public.
     * @return L'EntitySnapshot retiré (id=0 si non trouvé).
     */
    Partition::EntitySnapshot removeEntity(uint32_t publicId)
    {
        uint64_t chunkKey = _registry.getChunkKey(publicId);
        if (chunkKey == EntityRegistry::INVALID_CHUNK)
            return {};

        Partition *partition = _partitions.get(chunkKey);
        if (!partition)
            return {};

        auto snapshot = partition->removeEntityById(publicId);
        _registry.unregisterEntity(publicId);
        return snapshot;
    }

    /**
     * @brief Localise une entité par son ID public.
     * @param[out] outLocalIndex Index local dans le chunk (-1 si non trouvé).
     * @return Partition contenant l'entité, ou nullptr.
     */
    [[nodiscard]] Partition *findEntity(uint32_t publicId, int &outLocalIndex) const
    {
        uint64_t chunkKey = _registry.getChunkKey(publicId);
        if (chunkKey == EntityRegistry::INVALID_CHUNK)
        {
            outLocalIndex = -1;
            return nullptr;
        }

        Partition *partition = _partitions.get(chunkKey);
        if (!partition)
        {
            outLocalIndex = -1;
            return nullptr;
        }

        outLocalIndex = partition->findEntityIndex(publicId);
        return partition;
    }

    [[nodiscard]] uint64_t getEntityChunkKey(const uint32_t entityId) const noexcept
    {
        return _registry.getChunkKey(entityId);
    }

    [[nodiscard]] bool isEntityRegistered(uint32_t publicId) const noexcept
    {
        return _registry.isRegistered(publicId);
    }

    [[nodiscard]] const EntityRegistry &getRegistry() const noexcept { return _registry; }
    [[nodiscard]] EntityRegistry &getRegistry() noexcept { return _registry; }

    /**
     * @brief Itère sur tous les chunks actifs.
     */
    template <typename Func>
    void forEachChunk(Func &&func)
    {
        _partitions.forEach(std::forward<Func>(func));
    }

    /**
     * @brief Vérifie les bornes et réinsère les entités migrantes.
     * Appeler après la physique GPU (qui a déjà mis à jour positions/velocités).
     */
    void migrateEntities()
    {
        _transitQueue.clear();
        _partitions.forEach([&](Partition &partition) {
            partition.checkAndMigrate(_transitQueue);
        });
        for (const auto &entity : _transitQueue)
        {
            uint64_t key = getChunkKey(entity.position);
            if (Partition *partition = getOrCreateChunk(entity.position, key))
            {
                partition->addEntity(entity);
                _registry.updateChunkKey(entity.id, key);
            }
        }
    }

    /**
     * @brief Physique CPU + migration (chemin sans GPU).
     */
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
                _registry.updateChunkKey(entity.id, key);
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

        Partition newPartition({gridX, 0.f, gridZ}, _chunkSize);
        newPartition.reserve(256);
        return _partitions.insert(key, std::move(newPartition));
    }

private:
    static constexpr uint64_t WORLD_CAPACITY = 1ULL << 16ul;
    FlatAtomicsHashMap<Partition> _partitions;
    EntityRegistry _registry;
    std::vector<Partition::EntitySnapshot> _transitQueue;
    float _chunkSize;
};
