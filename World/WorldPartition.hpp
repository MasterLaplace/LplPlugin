/**
 * @file WorldPartition.hpp
 * @brief World Partitioning System for MMORPG
 *
 * Optimized spatial partitioning with FlatDynamicOctree per chunk:
 * - Dynamic loading/unloading based on player position
 * - Thread pool for asynchronous chunk loading
 * - Cache-friendly octrees (flat storage per chunk)
 * - Support for negative coordinates (Morton + bias)
 *
 * Performance:
 * - Chunk size: 255x255 (configurable)
 * - Load radius: 3x3 chunks around player
 * - Rebuild per chunk: ~0.02ms for 100-500 entities
 * - Total overhead: <1ms/frame for entire world
 *
 * Architecture:
 * ```
 * WorldPartition (global manager)
 *   └─> Partition (chunk 255x255)
 *         └─> FlatDynamicOctree (cache-friendly)
 *               └─> Entities (contiguous vector)
 * ```
 *
 * @author @MasterLaplace
 * @version 5.0 - Cleaned + Optimized
 * @date 2025-11-19
 */

#pragma once

#include "FlatDynamicOctree.hpp"
#if __has_include(<ankerl/unordered_dense.h>)
#    include <ankerl/unordered_dense.h>
#    define WP_HAS_ANKERL 1
#elif __has_include(<boost/unordered_flat_map.hpp>)
#    include <boost/unordered_flat_map.hpp>
#    define WP_HAS_BOOST_FLAT 1
#endif
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Optimizing::World {

/// Convert chunk coordinates (signed) to a Morton key with bias.
/// Supports coords roughly in [-2^(chunkBits-1), +2^(chunkBits-1)-1]
[[nodiscard]] inline uint64_t chunkMortonKey(const glm::ivec2 &coord, unsigned chunkBits = 21) noexcept
{
    const uint64_t bias = 1ULL << (chunkBits - 1);
    uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(coord.x) + static_cast<int64_t>(bias));
    uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>(coord.y) + static_cast<int64_t>(bias));
    return Morton::encode2D(static_cast<uint32_t>(ux), static_cast<uint32_t>(uz));
}

/**
 * @brief Simple thread pool for async chunk loading
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t threads = std::thread::hardware_concurrency()) : _stop(false)
    {
        _workers.reserve(threads);
        for (size_t i = 0; i < threads; ++i)
        {
            _workers.emplace_back([this] {
                while (true)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(_queueMutex);
                        _condition.wait(lock, [this] { return _stop || !_tasks.empty(); });

                        if (_stop && _tasks.empty())
                            return;

                        task = std::move(_tasks.front());
                        _tasks.pop();
                        _active.fetch_add(1, std::memory_order_relaxed);
                    }
                    task();
                    _active.fetch_sub(1, std::memory_order_relaxed);
                    _condition.notify_all();
                }
            });
        }
    }

    ~ThreadPool() { shutdown(); }

    /**
     * @brief Enqueue a callable for async execution
     *
     * Uses lambda capture instead of std::bind for performance
     * and correct move-only type support.
     */
    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            [func = std::forward<F>(f), ...capturedArgs = std::forward<Args>(args)]() mutable {
                return func(std::forward<Args>(capturedArgs)...);
            });

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(_queueMutex);
            if (_stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            _tasks.emplace([task]() { (*task)(); });
        }
        _condition.notify_one();
        return res;
    }

    void shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(_queueMutex);
            if (_stop)
                return;  // Guard against double shutdown
            _stop = true;
        }
        _condition.notify_all();

        for (std::thread &worker : _workers)
        {
            if (worker.joinable())
                worker.join();
        }
    }

    void waitIdle();

private:
    std::vector<std::thread> _workers;
    std::queue<std::function<void()>> _tasks;
    std::mutex _queueMutex;
    std::condition_variable _condition;
    bool _stop = false;
    std::atomic<size_t> _active{0};
};

/**
 * @brief Single partition (chunk) of the world
 *
 * Contains:
 * - Position & size (world space)
 * - FlatDynamicOctree for spatial queries
 * - Entities stored in contiguous vector
 * - Atomic load/unload state (thread-safe reads without mutex)
 */
template <typename Entity> class Partition {
public:
    Partition(const glm::vec3 &position, const glm::vec3 &size)
        : _position(position), _size(size),
          _octree(BoundingBox(position, position + size), 8, 32),
          _loaded(false)
    {
        _entities.reserve(512);
    }

    /// Add entity to partition (thread-safe)
    void addEntity(const Entity &entity)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _entities.push_back(entity);
    }

    /// Load partition (build octree from entities)
    void load()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_loaded.load(std::memory_order_relaxed) || _entities.empty())
            return;

        _octree.rebuild(_entities);
        _loaded.store(true, std::memory_order_release);
    }

    /// Unload partition (clear octree, keep entities)
    void unload()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_loaded.load(std::memory_order_relaxed))
            return;

        _octree.clear();
        _loaded.store(false, std::memory_order_release);
    }

    /// Update entities and rebuild octree
    void update()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_loaded.load(std::memory_order_relaxed) || _entities.empty())
            return;

        _octree.rebuild(_entities);
    }

    /// Query entities in bounding box (returns indices into _entities)
    [[nodiscard]] std::vector<uint32_t> query(const BoundingBox &box) const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_loaded.load(std::memory_order_relaxed))
            return {};
        return _octree.query(box);
    }

    [[nodiscard]] const std::vector<Entity> &getEntities() const { return _entities; }
    [[nodiscard]] const Entity &getEntity(uint32_t index) const { return _entities[index]; }

    /// Thread-safe loaded check (no mutex needed — atomic)
    [[nodiscard]] inline bool isLoaded() const noexcept
    {
        return _loaded.load(std::memory_order_acquire);
    }

    [[nodiscard]] inline BoundingBox getBounds() const noexcept
    {
        return BoundingBox(_position, _position + _size);
    }

    [[nodiscard]] inline const glm::vec3 &getPosition() const noexcept { return _position; }
    [[nodiscard]] inline const glm::vec3 &getSize() const noexcept { return _size; }
    [[nodiscard]] inline size_t entityCount() const noexcept { return _entities.size(); }

    void clear()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _entities.clear();
        _octree.clear();
        _loaded.store(false, std::memory_order_release);
    }

private:
    glm::vec3 _position;
    glm::vec3 _size;
    std::vector<Entity> _entities;
    FlatDynamicOctree _octree;
    mutable std::mutex _mutex;
    std::atomic<bool> _loaded;  // Atomic: safe to read without mutex
};

/**
 * @brief World Partition Manager for MMORPG
 *
 * Manages infinite 2D grid of chunks with dynamic loading.
 *
 * Usage:
 * ```cpp
 * WorldPartition<Entity> world(glm::vec3(255, 10000, 255));
 * world.insertEntities(allEntities);
 *
 * // Game loop
 * world.update(playerPosition);
 * auto nearby = world.query(searchBox);
 * ```
 */
template <typename Entity> class WorldPartition {
public:
    struct QueryResult {
        std::shared_ptr<Partition<Entity>> partition;
        std::vector<uint32_t> indices;
    };

    explicit WorldPartition(const glm::vec3 &chunkSize = glm::vec3(255.0f, 10000.0f, 255.0f), int loadRadius = 1,
                            size_t threadCount = std::thread::hardware_concurrency())
        : _chunkSize(chunkSize), _loadRadius(loadRadius), _chunkBits(21), _threadPool(threadCount)
    {
    }

    ~WorldPartition() { _threadPool.shutdown(); }

    /**
     * @brief Insert entities into world (batch)
     *
     * Distributes entities to appropriate chunks.
     * Cost: O(n) with hash map lookups.
     */
    void insertEntities(const std::vector<Entity> &entities)
    {
        std::lock_guard<std::mutex> lock(_mutex);
#if defined(WP_ENABLE_INSTRUMENTATION)
        using clk = std::chrono::high_resolution_clock;
        auto t_insert_start = clk::now();
#endif
        for (const auto &entity : entities)
        {
            glm::ivec2 chunkCoord = getChunkCoordinate(entity.position);
#if defined(WP_ENABLE_INSTRUMENTATION)
            auto t_key0 = clk::now();
#endif
            uint64_t key = chunkMortonKey(chunkCoord);
#if defined(WP_ENABLE_INSTRUMENTATION)
            auto t_key1 = clk::now();
            _instr_keyEncodeMs += std::chrono::duration<double, std::milli>(t_key1 - t_key0).count();
#endif

            auto it = _partitionsMorton.find(key);
            if (it == _partitionsMorton.end())
            {
                glm::vec3 chunkPos(chunkCoord.x * _chunkSize.x, 0.0f, chunkCoord.y * _chunkSize.z);
                auto partition = std::make_shared<Partition<Entity>>(chunkPos, _chunkSize);
                auto res = _partitionsMorton.emplace(key, partition);
                it = res.first;
                _sortedKeys.push_back(key);
                _sortedDirty = true;
            }

            it->second->addEntity(entity);
        }
#if defined(WP_ENABLE_INSTRUMENTATION)
        auto t_insert_end = clk::now();
        _instr_insertMs += std::chrono::duration<double, std::milli>(t_insert_end - t_insert_start).count();
        _instr_insertCount++;
#endif
    }

    /**
     * @brief Update world based on player position
     *
     * - Loads chunks within radius
     * - Unloads chunks outside radius
     * - Optionally rebuilds loaded chunks
     */
    void update(const glm::vec3 &playerPosition, bool rebuildChunks = false)
    {
        glm::ivec2 playerChunk = getChunkCoordinate(playerPosition);

        // Load chunks in radius (async)
        for (int x = -_loadRadius; x <= _loadRadius; ++x)
        {
            for (int z = -_loadRadius; z <= _loadRadius; ++z)
            {
                glm::ivec2 chunkCoord = playerChunk + glm::ivec2(x, z);
                loadChunk(chunkCoord);
            }
        }

        // Unload distant chunks
        std::vector<uint64_t> toUnloadKeys;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            for (const auto &[key, partition] : _partitionsMorton)
            {
                if (!partition->isLoaded())
                    continue;

                // Decode Morton key back to chunk coords
                uint32_t ux = 0, uz = 0;
                Morton::decode2D(static_cast<uint32_t>(key), ux, uz);
                const uint64_t bias = 1ULL << (_chunkBits - 1);
                int cx = static_cast<int>(static_cast<int64_t>(ux) - static_cast<int64_t>(bias));
                int cz = static_cast<int>(static_cast<int64_t>(uz) - static_cast<int64_t>(bias));

                int dx = std::abs(cx - playerChunk.x);
                int dz = std::abs(cz - playerChunk.y);
                if (dx > _loadRadius || dz > _loadRadius)
                    toUnloadKeys.push_back(key);
            }
        }

        for (uint64_t key : toUnloadKeys)
        {
            std::lock_guard<std::mutex> lock(_mutex);
            auto it = _partitionsMorton.find(key);
            if (it != _partitionsMorton.end())
                it->second->unload();
        }

        // Rebuild loaded chunks (if requested)
        if (rebuildChunks)
        {
            std::vector<std::shared_ptr<Partition<Entity>>> toUpdate;
            {
                std::lock_guard<std::mutex> lock(_mutex);
                toUpdate.reserve(_partitionsMorton.size());
                for (auto &[k, partition] : _partitionsMorton)
                {
                    if (partition->isLoaded())
                        toUpdate.push_back(partition);
                }
            }

#if defined(WP_ENABLE_INSTRUMENTATION)
            using clk = std::chrono::high_resolution_clock;
            auto t_update_start = clk::now();
#endif

            if (toUpdate.size() > 1)
            {
                size_t numWorkers = std::thread::hardware_concurrency();
                if (numWorkers == 0)
                    numWorkers = 2;
                numWorkers = std::min(numWorkers, toUpdate.size());
                size_t batchSize = (toUpdate.size() + numWorkers - 1) / numWorkers;

                std::vector<std::future<void>> futures;
                futures.reserve(numWorkers);
                for (size_t w = 0; w < numWorkers; ++w)
                {
                    size_t start = w * batchSize;
                    size_t end = std::min(start + batchSize, toUpdate.size());
                    if (start >= end)
                        continue;
                    futures.emplace_back(_threadPool.enqueue([
#if defined(WP_ENABLE_INSTRUMENTATION)
                        this,
#endif
                        &toUpdate, start, end]() {
                        for (size_t i = start; i < end; ++i)
                        {
#if defined(WP_ENABLE_INSTRUMENTATION)
                            using clk = std::chrono::high_resolution_clock;
                            auto t_p0 = clk::now();
                            toUpdate[i]->update();
                            auto t_p1 = clk::now();
                            _instr_partitionUpdateMs += std::chrono::duration<double, std::milli>(t_p1 - t_p0).count();
                            _instr_partitionUpdates++;
#else
                            toUpdate[i]->update();
#endif
                        }
                    }));
                }

                for (auto &fut : futures)
                    fut.get();
            }
            else
            {
                for (auto &partition : toUpdate)
                {
#if defined(WP_ENABLE_INSTRUMENTATION)
                    using clk = std::chrono::high_resolution_clock;
                    auto t_p0 = clk::now();
#endif
                    partition->update();
#if defined(WP_ENABLE_INSTRUMENTATION)
                    auto t_p1 = clk::now();
                    _instr_partitionUpdateMs += std::chrono::duration<double, std::milli>(t_p1 - t_p0).count();
                    _instr_partitionUpdates++;
#endif
                }
            }

#if defined(WP_ENABLE_INSTRUMENTATION)
            auto t_update_end = clk::now();
            _instr_updateMs += std::chrono::duration<double, std::milli>(t_update_end - t_update_start).count();
            _instr_updateCount++;
#endif
        }
    }

    /**
     * @brief Query entities in bounding box (all loaded chunks)
     *
     * Returns results grouped by partition.
     */
    [[nodiscard]] std::vector<QueryResult> query(const BoundingBox &searchBox) const
    {
        std::vector<QueryResult> results;
        std::lock_guard<std::mutex> lock(_mutex);

        for (const auto &[key, partition] : _partitionsMorton)
        {
            if (!partition->isLoaded())
                continue;

            if (partition->getBounds().overlaps(searchBox))
            {
                auto indices = partition->query(searchBox);
                if (!indices.empty())
                    results.push_back({partition, std::move(indices)});
            }
        }

        return results;
    }

    [[nodiscard]] std::shared_ptr<Partition<Entity>> getChunk(const glm::ivec2 &coord) const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        uint64_t key = chunkMortonKey(coord, _chunkBits);
        auto it = _partitionsMorton.find(key);
        return (it != _partitionsMorton.end()) ? it->second : nullptr;
    }

    [[nodiscard]] std::shared_ptr<Partition<Entity>> getChunkByMorton(uint64_t mortonKey) const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _partitionsMorton.find(mortonKey);
        return (it != _partitionsMorton.end()) ? it->second : nullptr;
    }

    [[nodiscard]] std::vector<std::shared_ptr<Partition<Entity>>> getLoadedChunks() const
    {
        std::vector<std::shared_ptr<Partition<Entity>>> loaded;
        std::lock_guard<std::mutex> lock(_mutex);

        for (const auto &[k, partition] : _partitionsMorton)
        {
            if (partition->isLoaded())
                loaded.push_back(partition);
        }

        return loaded;
    }

    /**
     * @brief Return partitions ordered by Morton key (cached — lazy sorted list)
     */
    [[nodiscard]] std::vector<std::shared_ptr<Partition<Entity>>> getAllPartitionsSortedByMorton() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_sortedDirty)
        {
            std::sort(_sortedKeys.begin(), _sortedKeys.end());
            _sortedDirty = false;
        }

        std::vector<std::shared_ptr<Partition<Entity>>> out;
        out.reserve(_sortedKeys.size());
        for (auto k : _sortedKeys)
        {
            auto it = _partitionsMorton.find(k);
            if (it != _partitionsMorton.end())
                out.push_back(it->second);
        }
        return out;
    }

    /**
     * @brief Return all partitions as pairs (mortonKey, partition)
     */
    [[nodiscard]] std::vector<std::pair<uint64_t, std::shared_ptr<Partition<Entity>>>> getAllPartitionsKeyed() const
    {
        std::vector<std::pair<uint64_t, std::shared_ptr<Partition<Entity>>>> out;
        std::lock_guard<std::mutex> lock(_mutex);
        out.reserve(_partitionsMorton.size());
        for (const auto &[k, partition] : _partitionsMorton)
            out.emplace_back(k, partition);
        return out;
    }

    struct Stats {
        size_t totalChunks = 0;
        size_t loadedChunks = 0;
        size_t totalEntities = 0;
        size_t loadedEntities = 0;
    };

    [[nodiscard]] Stats getStats() const
    {
        Stats stats;
        std::lock_guard<std::mutex> lock(_mutex);

        stats.totalChunks = _partitionsMorton.size();
        for (const auto &[k, partition] : _partitionsMorton)
        {
            stats.totalEntities += partition->entityCount();
            if (partition->isLoaded())
            {
                stats.loadedChunks++;
                stats.loadedEntities += partition->entityCount();
            }
        }

        return stats;
    }

#if defined(WP_ENABLE_INSTRUMENTATION)
    struct InstrStats {
        double insertMs = 0.0;
        double keyEncodeMs = 0.0;
        double updateMs = 0.0;
        double partitionUpdateMs = 0.0;
        size_t insertCount = 0;
        size_t updateCount = 0;
        size_t partitionUpdates = 0;
    };

    [[nodiscard]] InstrStats getInstrStats() const
    {
        InstrStats s;
        s.insertMs = _instr_insertMs;
        s.keyEncodeMs = _instr_keyEncodeMs;
        s.updateMs = _instr_updateMs;
        s.partitionUpdateMs = _instr_partitionUpdateMs;
        s.insertCount = _instr_insertCount;
        s.updateCount = _instr_updateCount;
        s.partitionUpdates = _instr_partitionUpdates;
        return s;
    }
#endif

    void clear()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _partitionsMorton.clear();
        _sortedKeys.clear();
        _sortedDirty = true;
    }

    /// Wait until pending async loads complete
    void waitForPendingLoads() { _threadPool.waitIdle(); }

    [[nodiscard]] inline const glm::vec3 &getChunkSize() const noexcept { return _chunkSize; }

private:
    [[nodiscard]] glm::ivec2 getChunkCoordinate(const glm::vec3 &position) const noexcept
    {
        return glm::ivec2(static_cast<int>(std::floor(position.x / _chunkSize.x)),
                          static_cast<int>(std::floor(position.z / _chunkSize.z)));
    }

    void loadChunk(const glm::ivec2 &coord)
    {
        std::shared_ptr<Partition<Entity>> partition;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            uint64_t key = chunkMortonKey(coord, _chunkBits);
            auto it = _partitionsMorton.find(key);
            if (it == _partitionsMorton.end())
                return;
            partition = it->second;
        }

        if (partition->isLoaded())
            return;

        _threadPool.enqueue([partition]() { partition->load(); });
    }

    glm::vec3 _chunkSize;
    int _loadRadius;
#if defined(WP_HAS_ANKERL)
    ankerl::unordered_dense::map<uint64_t, std::shared_ptr<Partition<Entity>>> _partitionsMorton;
#elif defined(WP_HAS_BOOST_FLAT)
    boost::unordered_flat_map<uint64_t, std::shared_ptr<Partition<Entity>>> _partitionsMorton;
#else
    std::unordered_map<uint64_t, std::shared_ptr<Partition<Entity>>> _partitionsMorton;
#endif
    unsigned _chunkBits;
    mutable std::vector<uint64_t> _sortedKeys;
    mutable bool _sortedDirty = true;
    mutable std::mutex _mutex;
    ThreadPool _threadPool;

#if defined(WP_ENABLE_INSTRUMENTATION)
    mutable std::atomic<double> _instr_insertMs{0.0};
    mutable std::atomic<double> _instr_keyEncodeMs{0.0};
    mutable std::atomic<double> _instr_updateMs{0.0};
    mutable std::atomic<double> _instr_partitionUpdateMs{0.0};
    mutable std::atomic<size_t> _instr_insertCount{0};
    mutable std::atomic<size_t> _instr_updateCount{0};
    mutable std::atomic<size_t> _instr_partitionUpdates{0};
#endif
};

} // namespace Optimizing::World
