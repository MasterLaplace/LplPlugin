#pragma once

#include "FlatAtomicsHashMap.hpp"
#include "Partition.hpp"
#include "EntityRegistry.hpp"
#include "PhysicsGPU.cuh"
#include "Morton.hpp"
#include "ThreadPool.hpp"
#include <cstring>

class WorldPartition {
public:
    WorldPartition() : _partitions(WORLD_CAPACITY), _chunkSize(1000.f), _writeIdx(0u), _threadPool()
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
    Partition::EntitySnapshot removeEntity(const uint32_t publicId)
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
    [[nodiscard]] Partition *findEntity(const uint32_t publicId, int &outLocalIndex) const
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

    [[nodiscard]] bool isEntityRegistered(const uint32_t publicId) const noexcept
    {
        return _registry.isRegistered(publicId);
    }

    [[nodiscard]] const EntityRegistry &getRegistry() const noexcept { return _registry; }
    [[nodiscard]] EntityRegistry &getRegistry() noexcept { return _registry; }

    /**
     * @brief Compte le nombre total d'entités dans tous les chunks.
     */
    [[nodiscard]] int getEntityCount() noexcept
    {
        int total = 0;
        _partitions.forEach([&](Partition &p) {
            total += static_cast<int>(p.getEntityCount());
        });
        return total;
    }

    /**
     * @brief Compte le nombre de chunks actifs.
     */
    [[nodiscard]] int getChunkCount() noexcept
    {
        int count = 0;
        _partitions.forEach([&](Partition &) { count++; });
        return count;
    }

    /**
     * @brief Compte le nombre d'entités dans la transit queue (migration en cours).
     */
    [[nodiscard]] size_t getTransitCount() const noexcept
    {
        return _transitQueue.size();
    }

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
        stepGPU(deltatime, wIdx);
#else
        stepCPU(deltatime, wIdx);
#endif

        flushTransitQueue();
        runGC();
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
        newPartition.setMortonKey(key);
        return _partitions.insert(key, std::move(newPartition));
    }

#if defined(__CUDACC__)
    /**
     * @brief Step physique pour le GPU.
     */
    void stepGPU(const float deltatime, const uint32_t wIdx)
    {
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
    }
#endif

    /**
     * @brief Step physique pour le fallback CPU.
     */
    void stepCPU(const float deltatime, const uint32_t wIdx)
    {
        const uint32_t nWorkers = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::vector<Partition::EntitySnapshot>> localTransits(nWorkers);

        _partitions.forEachParallel(_threadPool, [&](Partition &partition, uint32_t batchIdx) {
            partition.physicsTick(deltatime, localTransits[batchIdx % nWorkers], wIdx);
        });

        for (auto &lt : localTransits)
            _transitQueue.insert(_transitQueue.end(), lt.begin(), lt.end());
    }

    /**
     * @brief Réinsère les entités migrantes dans leurs nouveaux chunks après le step physique.
     */
    void flushTransitQueue()
    {
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
     * @brief Supprime les chunks vides pour libérer de la mémoire
     */
    void runGC()
    {
        if (++_gcCounter >= GC_INTERVAL_FRAMES)
        {
            _gcCounter = 0u;

            // Collecte des clés des chunks vides
            std::vector<uint64_t> emptyKeys;
            _partitions.forEach([&](Partition &p) {
                if (p.getEntityCount() == 0u && p.getMortonKey() != 0u)
                    emptyKeys.push_back(p.getMortonKey());
            });

            // Suppression hors de l'itération forEach
            for (uint64_t key : emptyKeys)
                _partitions.remove(key);
        }
    }

private:
    static constexpr uint64_t WORLD_CAPACITY = 1ULL << 16ul;
    static constexpr uint32_t GC_INTERVAL_FRAMES = 1u; ///< GC chaque frame (scan léger sur les chunks actifs)
    FlatAtomicsHashMap<Partition> _partitions;
    EntityRegistry _registry;
    std::vector<Partition::EntitySnapshot> _transitQueue;
    float _chunkSize;
    std::atomic<uint32_t> _writeIdx; ///< 0 ou 1 — index du write buffer global
    uint32_t _gcCounter = 0u;          ///< compteur de frames pour le GC périodique
    ThreadPool _threadPool;
};
