#pragma once

#include "FlatAtomicsHashMap.hpp"
#include "Partition.hpp"
#include "EntityRegistry.hpp"
#include "PhysicsGPU.cuh"
#include "Morton.hpp"
#include <cmath>
#include <cstring>

class WorldPartition {
public:
    WorldPartition() : _partitions(WORLD_CAPACITY), _chunkSize(255.f), _writeIdx(0u)
    {
        _transitQueue.reserve(1024u);
    }

    // ─── Double Buffer API ─────────────────────────────────────────

    /** @brief Index du write buffer courant (0 ou 1). */
    [[nodiscard]] uint32_t getWriteIdx() const noexcept
    {
        return _writeIdx.load(std::memory_order_acquire);
    }

    /** @brief Index du read buffer courant (stable pour le rendu). */
    [[nodiscard]] uint32_t getReadIdx() const noexcept
    {
        return _writeIdx.load(std::memory_order_acquire) ^ 1u;
    }

    /**
     * @brief Bascule le double buffer.
     *
     * 1. Copie les données hot (pos, vel, forces) du write buffer → read buffer
     *    (pour que le nouveau write buffer démarre avec l'état à jour).
     * 2. Toggle writeIdx (atomique, acquire-release).
     *
     * Après l'appel :
     *   - L'ancien write buffer devient le read buffer (snapshot stable pour le rendu)
     *   - Le nouveau write buffer contient une copie des dernières données
     */
    void swapBuffers()
    {
        uint32_t oldW = _writeIdx.load(std::memory_order_acquire);

        // Copie hot data : write → read (pour initialiser le nouveau write buffer)
        _partitions.forEach([oldW](Partition &p) {
            p.syncBuffers(oldW);
        });

        // Toggle atomique : l'ancien read buffer (maintenant synchronisé) devient write
        _writeIdx.fetch_xor(1u, std::memory_order_acq_rel);
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

        auto snapshot = partition->removeEntityById(publicId, getWriteIdx());
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
     * @brief Step physique + migration inter-chunk.
     *
     * Sélectionne automatiquement le backend :
     *   - GPU : cudaHostGetDevicePointer → launch_physics_kernel → gpu_sync
     *   - CPU : Partition::physicsTick (fallback si compilé sans nvcc)
     *
     * Écrit dans le write buffer. Appeler swapBuffers() ensuite.
     */
    void step(float deltatime)
    {
        uint32_t wIdx = getWriteIdx();
        _transitQueue.clear();

#if defined(__CUDACC__)
        // ─── GPU path ───────────────────────────────────────────────

#ifdef LPL_MONITORING
        gpu_timer_start();
#endif

        // Phase 1 : Lancement des kernels par chunk (pas de sync intermédiaire)
        _partitions.forEach([&](Partition &partition) {
            uint32_t count = static_cast<uint32_t>(partition.getEntityCount());
            if (count == 0u)
                return;

            Vec3  *d_pos = nullptr, *d_vel = nullptr, *d_forces = nullptr;
            float *d_masses = nullptr;

            CUDA_CHECK(cudaHostGetDevicePointer(&d_pos,    partition.getPositionsData(wIdx),  0));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_vel,    partition.getVelocitiesData(wIdx), 0));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_forces, partition.getForcesData(wIdx),     0));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_masses, partition.getMassesData(),          0));

            launch_physics_kernel(d_pos, d_vel, d_forces, d_masses, count, deltatime);
        });

        // Phase 2 : Sync unique après tous les kernels
        gpu_sync();

#ifdef LPL_MONITORING
        float ms = gpu_timer_stop();
        static int log_counter = 0;
        if (log_counter++ % 100 == 0)
            printf("[GPU] WorldPartition::step: %.3f ms\n", ms);
#endif

        // Phase 3 : Migration inter-chunk (CPU)
        _partitions.forEach([&](Partition &partition) {
            partition.checkAndMigrate(_transitQueue, wIdx);
        });

#else
        // ─── CPU fallback ───────────────────────────────────────────
        _partitions.forEach([&](Partition &partition){
            partition.physicsTick(deltatime, _transitQueue, wIdx);
        });
#endif

        // Réinsertion des entités migrantes
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
    std::atomic<uint32_t> _writeIdx; ///< 0 ou 1 — index du write buffer global
};
