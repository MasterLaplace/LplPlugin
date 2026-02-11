#pragma once

#include "FlatDynamicOctree.hpp"
#include "SpinLock.hpp"
#include "PinnedAllocator.hpp"
#include <vector>
#include <limits>
#include <unordered_map>

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
    Partition() noexcept : _bound({{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}}), _active(false), _octree(_bound) {}
    Partition(Vec3 position, float size) noexcept : _bound({
        {position.x, std::numeric_limits<float>::lowest(), position.z},
        {position.x + size, std::numeric_limits<float>::max(), position.z + size}
    }), _active(true), _octree(_bound) {}
    Partition(Partition &&other) noexcept :
        _ids(std::move(other._ids)),
        _positions(std::move(other._positions)),
        _rotations(std::move(other._rotations)),
        _velocities(std::move(other._velocities)),
        _masses(std::move(other._masses)),
        _forces(std::move(other._forces)),
        _sizes(std::move(other._sizes)),
        _health(std::move(other._health)),
        _idToLocal(std::move(other._idToLocal)),
        _bound(std::move(other._bound)),
        _active(other._active.load()),
        _octree(std::move(other._octree)) {}

    Partition &operator=(Partition &&other) noexcept {
        if (this != &other) {
            _ids = std::move(other._ids);
            _positions = std::move(other._positions);
            _rotations = std::move(other._rotations);
            _velocities = std::move(other._velocities);
            _masses = std::move(other._masses);
            _forces = std::move(other._forces);
            _sizes = std::move(other._sizes);
            _health = std::move(other._health);
            _idToLocal = std::move(other._idToLocal);
            _bound = std::move(other._bound);
            _active = other._active.load();
            _octree = std::move(other._octree);
        }
        return *this;
    }

    uint32_t addEntity(const EntitySnapshot &entity)
    {
        LocalGuard guard(_locker);
        uint32_t localIndex = static_cast<uint32_t>(_ids.size());
        _ids.push_back(entity.id);
        _positions.push_back(entity.position);
        _rotations.push_back(entity.rotation);
        _velocities.push_back(entity.velocity);
        _masses.push_back(entity.mass);
        _forces.push_back(entity.force);
        _sizes.push_back(entity.size);
        _health.push_back(entity.health);
        _idToLocal[entity.id] = localIndex;
        return localIndex;
    }

    [[nodiscard]] EntityRef getEntity(const size_t index) noexcept {return EntityRef{
        _positions[index],
        _rotations[index],
        _velocities[index],
        _masses[index],
        _forces[index],
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
        auto it = _idToLocal.find(id);
        if (it != _idToLocal.end())
            return static_cast<int>(it->second);
        return -1;
    }

    /**
     * @brief Retire une entité par son ID. Swap-and-pop O(1).
     * @return EntitySnapshot de l'entité retirée (id=0 si non trouvée).
     */
    EntitySnapshot removeEntityById(uint32_t entityId)
    {
        LocalGuard guard(_locker);
        auto it = _idToLocal.find(entityId);
        if (it == _idToLocal.end())
            return {};

        uint32_t index = it->second;
        EntitySnapshot removed = {
            _ids[index], _positions[index], _rotations[index],
            _velocities[index], _masses[index], _forces[index], _sizes[index],
            _health[index]
        };

        _idToLocal.erase(it);
        size_t last = _ids.size() - 1;

        if (index != last)
        {
            uint32_t movedId = _ids[last];
            _ids[index] = _ids[last];
            _positions[index] = _positions[last];
            _rotations[index] = _rotations[last];
            _velocities[index] = _velocities[last];
            _masses[index] = _masses[last];
            _forces[index] = _forces[last];
            _sizes[index] = _sizes[last];
            _health[index] = _health[last];
            _idToLocal[movedId] = index;
        }

        _ids.pop_back();
        _positions.pop_back();
        _rotations.pop_back();
        _velocities.pop_back();
        _masses.pop_back();
        _forces.pop_back();
        _sizes.pop_back();
        _health.pop_back();

        return removed;
    }

    /**
     * @brief Tick physique + détection de migration.
     * Deux passes pour éviter le double-tick d'entités swappées :
     *   Pass 1: Intégration physique (forward)
     *   Pass 2: Bounds check + swap-and-pop (backward)
     */
    void physicsTick(const float deltatime, std::vector<EntitySnapshot> &out_migrating) noexcept
    {
        LocalGuard guard(_locker);

        // Pass 1: Intégration physique (toutes les entités, une seule fois)
        for (size_t i = 0; i < _ids.size(); ++i)
        {
            _forces[i] = Vec3{0.0f, -9.81f * _masses[i], 0.0f};
            if (_masses[i] > 0.0001f)
            {
                Vec3 acceleration = _forces[i] * (1.0f / _masses[i]);
                _velocities[i] += acceleration * deltatime;
            }
            _positions[i] += _velocities[i] * deltatime;
        }

        // Pass 2: Bounds check + migration (backward pour éviter invalidation d'index)
        for (int64_t i = static_cast<int64_t>(_ids.size()) - 1; i >= 0; --i)
        {
            if (_bound.contains(_positions[i]))
                continue;

            out_migrating.push_back({
                _ids[i], _positions[i], _rotations[i],
                _velocities[i], _masses[i], _forces[i], _sizes[i], _health[i]
            });

            _idToLocal.erase(_ids[i]);
            size_t last = _ids.size() - 1;

            if (static_cast<size_t>(i) != last)
            {
                uint32_t movedId = _ids[last];
                _ids[i] = _ids[last];
                _positions[i] = _positions[last];
                _rotations[i] = _rotations[last];
                _velocities[i] = _velocities[last];
                _masses[i] = _masses[last];
                _forces[i] = _forces[last];
                _sizes[i] = _sizes[last];
                _health[i] = _health[last];
                _idToLocal[movedId] = static_cast<uint32_t>(i);
            }

            _ids.pop_back();
            _positions.pop_back();
            _rotations.pop_back();
            _velocities.pop_back();
            _masses.pop_back();
            _forces.pop_back();
            _sizes.pop_back();
            _health.pop_back();
        }
    }

    // ─── Raw SoA accessors (for GPU kernels) ──────────────────────

    [[nodiscard]] Vec3  *getPositionsData()  noexcept { return _positions.data(); }
    [[nodiscard]] Vec3  *getVelocitiesData() noexcept { return _velocities.data(); }
    [[nodiscard]] Vec3  *getForcesData()     noexcept { return _forces.data(); }
    [[nodiscard]] float *getMassesData()     noexcept { return _masses.data(); }
    [[nodiscard]] int32_t *getHealthData()    noexcept { return _health.data(); }
    [[nodiscard]] const BoundaryBox &getBound() const noexcept { return _bound; }

    // ─── Component Setters (for network updates) ──────────────────

    void setPosition(uint32_t localIdx, Vec3 pos) noexcept { _positions[localIdx] = pos; }
    void setVelocity(uint32_t localIdx, Vec3 vel) noexcept { _velocities[localIdx] = vel; }
    void setMass(uint32_t localIdx, float m)       noexcept { _masses[localIdx] = m; }
    void setHealth(uint32_t localIdx, int32_t hp)  noexcept { _health[localIdx] = hp; }

    /**
     * @brief Pré-alloue la capacité des vecteurs SoA.
     */
    void reserve(size_t capacity)
    {
        _ids.reserve(capacity);
        _positions.reserve(capacity);
        _rotations.reserve(capacity);
        _velocities.reserve(capacity);
        _masses.reserve(capacity);
        _forces.reserve(capacity);
        _sizes.reserve(capacity);
        _health.reserve(capacity);
    }

    /**
     * @brief Vérifie les bornes et extrait les entités migrantes.
     * Utilisé après la physique GPU (qui a déjà mis à jour positions/velocités).
     * Ne fait PAS d'intégration physique — uniquement bounds-check + swap-and-pop.
     * Itère en arrière pour éviter l'invalidation d'index lors du swap-and-pop.
     */
    void checkAndMigrate(std::vector<EntitySnapshot> &out_migrating) noexcept
    {
        LocalGuard guard(_locker);

        for (int64_t i = static_cast<int64_t>(_ids.size()) - 1; i >= 0; --i)
        {
            if (_bound.contains(_positions[i]))
                continue;

            out_migrating.push_back({
                _ids[i], _positions[i], _rotations[i],
                _velocities[i], _masses[i], _forces[i], _sizes[i], _health[i]
            });

            _idToLocal.erase(_ids[i]);
            size_t last = _ids.size() - 1;

            if (static_cast<size_t>(i) != last)
            {
                uint32_t movedId = _ids[last];
                _ids[i] = _ids[last];
                _positions[i] = _positions[last];
                _rotations[i] = _rotations[last];
                _velocities[i] = _velocities[last];
                _masses[i] = _masses[last];
                _forces[i] = _forces[last];
                _sizes[i] = _sizes[last];
                _health[i] = _health[last];
                _idToLocal[movedId] = static_cast<uint32_t>(i);
            }

            _ids.pop_back();
            _positions.pop_back();
            _rotations.pop_back();
            _velocities.pop_back();
            _masses.pop_back();
            _forces.pop_back();
            _sizes.pop_back();
            _health.pop_back();
        }
    }

    // ─── Spatial Index ─────────────────────────────────────────────

    void updateSpatialIndex()
    {
        _octree.rebuild(_positions.size(), [&](const uint32_t index){
            return BoundaryBox{_positions[index] - (_sizes[index] * 0.5f), _positions[index] + (_sizes[index] * 0.5f)};
        });
    }

    template <typename Func>
    void queryRegion(const BoundaryBox &area, Func &&callback)
    {
        _octree.query(area, std::forward<Func>(callback));
    }

private:
    std::vector<uint32_t, PinnedAllocator<uint32_t>> _ids;
    std::vector<Vec3, PinnedAllocator<Vec3>> _positions;
    std::vector<Quat, PinnedAllocator<Quat>> _rotations;
    std::vector<Vec3, PinnedAllocator<Vec3>> _velocities;
    std::vector<float, PinnedAllocator<float>> _masses;
    std::vector<Vec3, PinnedAllocator<Vec3>> _forces;
    std::vector<Vec3, PinnedAllocator<Vec3>> _sizes;
    std::vector<int32_t, PinnedAllocator<int32_t>> _health;
    std::unordered_map<uint32_t, uint32_t> _idToLocal; ///< entityId → local index O(1)
    BoundaryBox _bound;
    SpinLock _locker;
    std::atomic<bool> _active;
    FlatDynamicOctree _octree;
};
