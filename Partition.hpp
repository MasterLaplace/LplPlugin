#pragma once

#include "FlatDynamicOctree.hpp"
#include "SpinLock.hpp"
#include "PinnedAllocator.hpp"
#include <vector>
#include <limits>
#include <unordered_map>
#include <cstring>

/**
 * @brief Partition (chunk) ECS avec stockage SoA et double buffering des données hot.
 *
 * Données hot (double-buffered, indexées par writeIdx) :
 *   positions[2], velocities[2], forces[2]
 *
 * Données cold (buffer unique) :
 *   ids, rotations, masses, sizes, health
 *
 * Le double buffering permet une lecture concurrente sans lock (reader↔writer) :
 *   - Les writers (physique GPU, réseau) écrivent dans buf[writeIdx]
 *   - Les readers (rendu) lisent depuis buf[writeIdx ^ 1] (snapshot stable)
 *   - swapBuffers() copie le contenu du write buffer vers le nouveau write buffer
 *     puis bascule writeIdx, rendant les résultats visibles aux readers.
 *
 * Le SpinLock protège les mutations structurelles (writer↔writer) :
 *   - addEntity / removeEntityById modifient la taille des vecteurs (push_back, swap-and-pop)
 *   - physicsTick / checkAndMigrate font de la migration (swap-and-pop concurrent)
 *   Sans ce lock, deux writers simultanés (ex: réseau + physique) corrompraient les vecteurs.
 */
class Partition {
public:
    struct EntityRef {
        Vec3 &position;
        Quat &rotation;
        Vec3 &velocity;
        float &mass;
        Vec3 &force;
        Vec3 size;
        int32_t &health;
    };

    struct EntitySnapshot {
        uint32_t id;
        Vec3 position;
        Quat rotation;
        Vec3 velocity;
        float mass;
        Vec3 force;
        Vec3 size;
        int32_t health = 100;
    };

public:
    Partition() noexcept : _sparseCapacity(0), _bound({{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}}), _active(false), _octree(_bound) {}
    Partition(Vec3 position, float size) noexcept : _sparseCapacity(0), _bound({
        {position.x, std::numeric_limits<float>::lowest(), position.z},
        {position.x + size, std::numeric_limits<float>::max(), position.z + size}
    }), _active(true), _octree(_bound) {}

    Partition(Partition &&other) noexcept :
        _sparseToLocal(std::move(other._sparseToLocal)),
        _sparseCapacity(other._sparseCapacity),
        _ids(std::move(other._ids)),
        _rotations(std::move(other._rotations)),
        _masses(std::move(other._masses)),
        _sizes(std::move(other._sizes)),
        _health(std::move(other._health)),
        _positions{std::move(other._positions[0]), std::move(other._positions[1])},
        _velocities{std::move(other._velocities[0]), std::move(other._velocities[1])},
        _forces{std::move(other._forces[0]), std::move(other._forces[1])},
        _bound(std::move(other._bound)),
        _active(other._active.load()),
        _octree(std::move(other._octree)) {}

    Partition &operator=(Partition &&other) noexcept {
        if (this != &other) {
            _ids = std::move(other._ids);
            for (uint32_t b = 0; b < 2; ++b)
            {
                _positions[b] = std::move(other._positions[b]);
                _velocities[b] = std::move(other._velocities[b]);
                _forces[b] = std::move(other._forces[b]);
            }
            _rotations = std::move(other._rotations);
            _masses = std::move(other._masses);
            _sizes = std::move(other._sizes);
            _health = std::move(other._health);
            _sparseToLocal = std::move(other._sparseToLocal);
            _bound = std::move(other._bound);
            _sparseCapacity = other._sparseCapacity;
            _active = other._active.load();
            _octree = std::move(other._octree);
        }
        return *this;
    }

    /**
     * @brief Ajoute une entité au chunk.
     * Écrit dans les DEUX buffers pour que read et write soient cohérents dès l'insertion.
     */
    uint32_t addEntity(const EntitySnapshot &entity)
    {
        LocalGuard guard(_locker);
        uint32_t localIndex = static_cast<uint32_t>(_ids.size());
        _ids.push_back(entity.id);

        // Hot data → écriture dans les deux buffers (read + write initialisés identiquement)
        for (uint32_t b = 0; b < 2; ++b)
        {
            _positions[b].push_back(entity.position);
            _velocities[b].push_back(entity.velocity);
            _forces[b].push_back(entity.force);
        }

        // Cold data → buffer unique
        _rotations.push_back(entity.rotation);
        _masses.push_back(entity.mass);
        _sizes.push_back(entity.size);
        _health.push_back(entity.health);

        // Lazy allocation du sparse set : redimensionner si nécessaire
        if (entity.id >= _sparseCapacity) {
            uint32_t newCapacity = entity.id + (entity.id >> 1);
            newCapacity = std::max(newCapacity, 1024u);     // Min 4KB
            newCapacity = std::min(newCapacity, 262144u);   // Max 1MB (65k IDs)
            newCapacity = 1u << (32u - __builtin_clz(newCapacity - 1)); // Align puissance de 2
            _sparseToLocal.resize(newCapacity, INVALID_INDEX);
            _sparseCapacity = newCapacity;
        }
        if (entity.id < _sparseCapacity)
            _sparseToLocal[entity.id] = localIndex;
        return localIndex;
    }

    /**
     * @brief Accède à une entité par index local.
     * @param bufIdx Index de buffer (0 ou 1) — utiliser readIdx pour le rendu, writeIdx pour l'écriture.
     */
    [[nodiscard]] EntityRef getEntity(const size_t index, uint32_t bufIdx) noexcept {return EntityRef{
        _positions[bufIdx][index],
        _rotations[index],
        _velocities[bufIdx][index],
        _masses[index],
        _forces[bufIdx][index],
        _sizes[index],
        _health[index]
    }; }

    [[nodiscard]] size_t getEntityCount() const noexcept {
        return _ids.size();
    }

    [[nodiscard]] uint32_t getEntityId(const size_t index) const noexcept {
        return _ids[index];
    }

    [[nodiscard]] int findEntityIndex(const uint32_t id) const noexcept {
        if (id >= _sparseCapacity || _sparseToLocal[id] == INVALID_INDEX)
            return -1;
        return static_cast<int>(_sparseToLocal[id]);
    }

    /**
     * @brief Retire une entité par son ID. Swap-and-pop O(1) sur les deux buffers.
     * @param writeIdx Index du write buffer (pour le snapshot retourné).
     * @return EntitySnapshot de l'entité retirée (id=0 si non trouvée).
     */
    EntitySnapshot removeEntityById(uint32_t entityId, uint32_t writeIdx)
    {
        LocalGuard guard(_locker);
        if (entityId >= _sparseCapacity || _sparseToLocal[entityId] == INVALID_INDEX)
            return {};

        uint32_t index = _sparseToLocal[entityId];

        // Snapshot depuis le write buffer (données les plus récentes)
        EntitySnapshot removed = {
            _ids[index], _positions[writeIdx][index], _rotations[index],
            _velocities[writeIdx][index], _masses[index], _forces[writeIdx][index],
            _sizes[index], _health[index]
        };

        _sparseToLocal[entityId] = INVALID_INDEX;
        size_t last = _ids.size() - 1;

        if (index != last)
        {
            uint32_t movedId = _ids[last];
            _ids[index] = _ids[last];

            // Swap-and-pop les DEUX buffers hot
            for (uint32_t b = 0; b < 2; ++b)
            {
                _positions[b][index] = _positions[b][last];
                _velocities[b][index] = _velocities[b][last];
                _forces[b][index] = _forces[b][last];
            }

            // Cold data
            _rotations[index] = _rotations[last];
            _masses[index] = _masses[last];
            _sizes[index] = _sizes[last];
            _health[index] = _health[last];
            if (movedId < _sparseCapacity)
                _sparseToLocal[movedId] = index;
        }

        _ids.pop_back();
        for (uint32_t b = 0; b < 2; ++b)
        {
            _positions[b].pop_back();
            _velocities[b].pop_back();
            _forces[b].pop_back();
        }
        _rotations.pop_back();
        _masses.pop_back();
        _sizes.pop_back();
        _health.pop_back();

        return removed;
    }

    /**
     * @brief Tick physique CPU + détection de migration.
     *
     * Deux passes pour éviter le double-tick d'entités swappées :
     *   Pass 1 : Intégration physique (in-place sur le write buffer)
     *   Pass 2 : Bounds check + swap-and-pop (backward, les deux buffers)
     *
     * @param writeIdx Index du write buffer où la physique écrit.
     */
    void physicsTick(const float deltatime, std::vector<EntitySnapshot> &out_migrating, uint32_t writeIdx) noexcept
    {
        LocalGuard guard(_locker);

        for (size_t i = 0; i < _ids.size(); ++i)
        {
            _forces[writeIdx][i] += Vec3{0.0f, -9.81f * _masses[i], 0.0f};
            if (_masses[i] > 0.0001f)
            {
                Vec3 acceleration = _forces[writeIdx][i] * (1.0f / _masses[i]);
                _velocities[writeIdx][i] += acceleration * deltatime;
            }
            _positions[writeIdx][i] += _velocities[writeIdx][i] * deltatime;
            if (_positions[writeIdx][i].y < 0.f)
                _positions[writeIdx][i] = Vec3{_positions[writeIdx][i].x, 0.f, _positions[writeIdx][i].z};
            _forces[writeIdx][i] = Vec3{0.f, 0.f, 0.f};
        }

        _octree.rebuild(_positions[writeIdx].size(), [&](const uint32_t index){
            return BoundaryBox{
                _positions[writeIdx][index] - (_sizes[index] * 0.5f),
                _positions[writeIdx][index] + (_sizes[index] * 0.5f)
            };
        });

        for (size_t i = 0; i < _ids.size(); ++i)
        {
            _octree.query(BoundaryBox{
                _positions[writeIdx][i] - (_sizes[i] * 0.5f),
                _positions[writeIdx][i] + (_sizes[i] * 0.5f)
            }, [&](const uint32_t index) {
                if (i == index) return;
                Vec3 delta = _positions[writeIdx][i] - _positions[writeIdx][index];
                Vec3 absDelta = Vec3{std::fabs(delta.x), std::fabs(delta.y), std::fabs(delta.z)};
                Vec3 minDist = (_sizes[i] * 0.5f) + (_sizes[index] * 0.5f);

                if (absDelta < minDist)
                {
                    float distSq = delta.magnitudeSquared();

                    if (distSq < 0.0001f)
                    {
                        delta = Vec3{0.0f, 1.0f, 0.0f};
                        distSq = 1.0f;
                    }

                    Vec3 repulsion = delta * (50.0f / (distSq + 0.1f));
                    _forces[writeIdx][i] += repulsion;
                }
            });
        }

        for (int64_t i = static_cast<int64_t>(_ids.size()) - 1; i >= 0; --i)
        {
            if (_bound.contains(_positions[writeIdx][i]))
                continue;

            out_migrating.push_back({
                _ids[i], _positions[writeIdx][i], _rotations[i],
                _velocities[writeIdx][i], _masses[i], _forces[writeIdx][i], _sizes[i], _health[i]
            });

            uint32_t removedId = _ids[i];
            if (removedId < _sparseCapacity)
                _sparseToLocal[removedId] = INVALID_INDEX;
            size_t last = _ids.size() - 1;

            if (static_cast<size_t>(i) != last)
            {
                uint32_t movedId = _ids[last];
                _ids[i] = _ids[last];
                for (uint32_t b = 0; b < 2; ++b)
                {
                    _positions[b][i] = _positions[b][last];
                    _velocities[b][i] = _velocities[b][last];
                    _forces[b][i] = _forces[b][last];
                }
                _rotations[i] = _rotations[last];
                _masses[i] = _masses[last];
                _sizes[i] = _sizes[last];
                _health[i] = _health[last];
                if (movedId < _sparseCapacity)
                    _sparseToLocal[movedId] = static_cast<uint32_t>(i);
            }

            _ids.pop_back();
            for (uint32_t b = 0; b < 2; ++b)
            {
                _positions[b].pop_back();
                _velocities[b].pop_back();
                _forces[b].pop_back();
            }
            _rotations.pop_back();
            _masses.pop_back();
            _sizes.pop_back();
            _health.pop_back();
        }
    }

    // ─── Raw SoA Accessors (double-buffered) ──────────────────────

    [[nodiscard]] Vec3  *getPositionsData(uint32_t bufIdx)  noexcept { return _positions[bufIdx].data(); }
    [[nodiscard]] Vec3  *getVelocitiesData(uint32_t bufIdx) noexcept { return _velocities[bufIdx].data(); }
    [[nodiscard]] Vec3  *getForcesData(uint32_t bufIdx)     noexcept { return _forces[bufIdx].data(); }

    // ─── Raw SoA Accessors (cold, single buffer) ─────────────────

    [[nodiscard]] float   *getMassesData()  noexcept { return _masses.data(); }
    [[nodiscard]] int32_t *getHealthData()  noexcept { return _health.data(); }
    [[nodiscard]] const BoundaryBox &getBound() const noexcept { return _bound; }

    // ─── Component Setters ────────────────────────────────────────
    // Hot data : écrivent dans le write buffer uniquement.
    // Cold data : écrivent dans le buffer unique.

    void setPosition(uint32_t localIdx, Vec3 pos, uint32_t writeIdx) noexcept { _positions[writeIdx][localIdx] = pos; }
    void setVelocity(uint32_t localIdx, Vec3 vel, uint32_t writeIdx) noexcept { _velocities[writeIdx][localIdx] = vel; }
    void setMass(uint32_t localIdx, float m)       noexcept { _masses[localIdx] = m; }
    void setSize(uint32_t localIdx, Vec3 s)         noexcept { _sizes[localIdx] = s; }
    void setHealth(uint32_t localIdx, int32_t hp)  noexcept { _health[localIdx] = hp; }

    /**
     * @brief Pré-alloue la capacité pour les deux buffers + les vecteurs cold.
     */
    void reserve(size_t capacity)
    {
        _ids.reserve(capacity);
        for (uint32_t b = 0; b < 2; ++b)
        {
            _positions[b].reserve(capacity);
            _velocities[b].reserve(capacity);
            _forces[b].reserve(capacity);
        }
        _rotations.reserve(capacity);
        _masses.reserve(capacity);
        _sizes.reserve(capacity);
        _health.reserve(capacity);
    }

    /**
     * @brief Vérifie les bornes et extrait les entités migrantes.
     * Utilisé après la physique GPU. Le write buffer contient les positions mises à jour.
     * Swap-and-pop sur les DEUX buffers pour maintenir la cohérence.
     *
     * @param writeIdx Index du write buffer (positions à checker).
     */
    void checkAndMigrate(std::vector<EntitySnapshot> &out_migrating, uint32_t writeIdx) noexcept
    {
        LocalGuard guard(_locker);

        for (int64_t i = static_cast<int64_t>(_ids.size()) - 1; i >= 0; --i)
        {
            if (_bound.contains(_positions[writeIdx][i]))
                continue;

            out_migrating.push_back({
                _ids[i], _positions[writeIdx][i], _rotations[i],
                _velocities[writeIdx][i], _masses[i], _forces[writeIdx][i], _sizes[i], _health[i]
            });

            uint32_t removedId = _ids[i];
            if (removedId < _sparseCapacity)
                _sparseToLocal[removedId] = INVALID_INDEX;
            size_t last = _ids.size() - 1;

            if (static_cast<size_t>(i) != last)
            {
                uint32_t movedId = _ids[last];
                _ids[i] = _ids[last];
                for (uint32_t b = 0; b < 2; ++b)
                {
                    _positions[b][i] = _positions[b][last];
                    _velocities[b][i] = _velocities[b][last];
                    _forces[b][i] = _forces[b][last];
                }
                _rotations[i] = _rotations[last];
                _masses[i] = _masses[last];
                _sizes[i] = _sizes[last];
                _health[i] = _health[last];
                if (movedId < _sparseCapacity)
                    _sparseToLocal[movedId] = static_cast<uint32_t>(i);
            }

            _ids.pop_back();
            for (uint32_t b = 0; b < 2; ++b)
            {
                _positions[b].pop_back();
                _velocities[b].pop_back();
                _forces[b].pop_back();
            }
            _rotations.pop_back();
            _masses.pop_back();
            _sizes.pop_back();
            _health.pop_back();
        }
    }

    /**
     * @brief Copie les données hot du write buffer vers l'autre buffer.
     * Appelé par WorldPartition::swapBuffers() AVANT le toggle de writeIdx.
     * Après copie + toggle, le nouveau write buffer a les données à jour.
     *
     * @param writeIdx Index actuel du write buffer (source de la copie).
     */
    void syncBuffers(uint32_t writeIdx) noexcept
    {
        uint32_t readIdx = writeIdx ^ 1u;
        size_t n = _ids.size();
        if (n == 0)
            return;

        std::memcpy(_positions[readIdx].data(),  _positions[writeIdx].data(),  n * sizeof(Vec3));
        std::memcpy(_velocities[readIdx].data(), _velocities[writeIdx].data(), n * sizeof(Vec3));
        std::memcpy(_forces[readIdx].data(),     _forces[writeIdx].data(),     n * sizeof(Vec3));
    }

    // ─── Spatial Index ─────────────────────────────────────────────

    void updateSpatialIndex(uint32_t bufIdx = 0)
    {
        _octree.rebuild(_positions[bufIdx].size(), [&](const uint32_t index){
            return BoundaryBox{
                _positions[bufIdx][index] - (_sizes[index] * 0.5f),
                _positions[bufIdx][index] + (_sizes[index] * 0.5f)
            };
        });
    }

    template <typename Func>
    void queryRegion(const BoundaryBox &area, Func &&callback)
    {
        _octree.query(area, std::forward<Func>(callback));
    }

private:
    // ─── Chunk metadata (order matters for ctor initialization) ──────────
    static constexpr uint32_t INVALID_INDEX = UINT32_MAX;
    std::vector<uint32_t> _sparseToLocal;     ///< entityId → local index (sparse set, O(1) guaranteed, lazy-allocated)
    uint32_t _sparseCapacity;                  ///< capacité actuelle du sparse set (0 initially)

    // ─── Cold data (single buffer) ────────────────────────────────
    std::vector<uint32_t, PinnedAllocator<uint32_t>> _ids;
    std::vector<Quat, PinnedAllocator<Quat>> _rotations;
    std::vector<float, PinnedAllocator<float>> _masses;
    std::vector<Vec3, PinnedAllocator<Vec3>> _sizes;
    std::vector<int32_t, PinnedAllocator<int32_t>> _health;

    // ─── Hot data (double-buffered) ───────────────────────────────
    std::vector<Vec3, PinnedAllocator<Vec3>> _positions[2];
    std::vector<Vec3, PinnedAllocator<Vec3>> _velocities[2];
    std::vector<Vec3, PinnedAllocator<Vec3>> _forces[2];

    // ─── Synchronization ──────────────────────────────────────────
    BoundaryBox _bound;
    SpinLock _locker;
    std::atomic<bool> _active;
    FlatDynamicOctree _octree;
};
