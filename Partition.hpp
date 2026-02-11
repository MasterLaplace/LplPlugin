#pragma once

#include "FlatDynamicOctree.hpp"
#include "SpinLock.hpp"
#include <vector>
#include <limits>

class Partition {
public:
    struct EntityRef {
        Vec3 &position;
        Quat &rotation;
        Vec3 &velocity;
        float &mass;
        Vec3 &force;
        Vec3 size;
    };

    struct EntitySnapshot {
        uint32_t id;
        Vec3 position;
        Quat rotation;
        Vec3 velocity;
        float mass;
        Vec3 force;
        Vec3 size;
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
            _bound = std::move(other._bound);
            _active = other._active.load();
            _octree = std::move(other._octree);
        }
        return *this;
    }

    void addEntity(const EntitySnapshot &entity)
    {
        LocalGuard guard(_locker);
        _ids.push_back(entity.id);
        _positions.push_back(entity.position);
        _rotations.push_back(entity.rotation);
        _velocities.push_back(entity.velocity);
        _masses.push_back(entity.mass);
        _forces.push_back(entity.force);
        _sizes.push_back(entity.size);
    }

    [[nodiscard]] EntityRef getEntity(const size_t index) noexcept {return EntityRef{
        _positions[index],
        _rotations[index],
        _velocities[index],
        _masses[index],
        _forces[index],
        _sizes[index]
    }; }

    [[nodiscard]] size_t getEntityCount() const noexcept {
        return _ids.size();
    }

    void physicsTick(float deltatime, std::vector<EntitySnapshot> &out_migrating) noexcept
    {
        LocalGuard guard(_locker);

        for (uint64_t index = 0u; index < _ids.size(); ++index)
        {
            _forces[index] = Vec3{0.0f, -9.81f * _masses[index], 0.0f};
            if (_masses[index] > 0.0001f)
            {
                Vec3 acceleration = _forces[index] * (1.0f / _masses[index]);
                _velocities[index] += acceleration * deltatime;
            }
            _positions[index] += _velocities[index] * deltatime;

            if (_bound.contains(_positions[index]))
                continue;

            out_migrating.push_back({
                _ids[index],
                _positions[index],
                _rotations[index],
                _velocities[index],
                _masses[index],
                _forces[index],
                _sizes[index]
            });

            size_t last = _ids.size() - 1ul;

            if (index != last)
            {
                _ids[index] = _ids[last];
                _positions[index] = _positions[last];
                _rotations[index] = _rotations[last];
                _velocities[index] = _velocities[last];
                _masses[index] = _masses[last];
                _forces[index] = _forces[last];
                _sizes[index] = _sizes[last];
            }

            _ids.pop_back();
            _positions.pop_back();
            _rotations.pop_back();
            _velocities.pop_back();
            _masses.pop_back();
            _forces.pop_back();
            _sizes.pop_back();

            --index;
        }
    }

    void updateSpatialIndex()
    {
        _octree.rebuild(_positions.size(), [&](uint32_t index){
            return BoundaryBox{_positions[index] - (_sizes[index] * 0.5f), _positions[index] + (_sizes[index] * 0.5f)};
        });
    }

    template <typename Func>
    void queryRegion(const BoundaryBox &area, Func &&callback)
    {
        _octree.query(area, std::forward<Func>(callback));
    }

private:
    std::vector<uint32_t> _ids;
    std::vector<Vec3> _positions;
    std::vector<Quat> _rotations;
    std::vector<Vec3> _velocities;
    std::vector<float> _masses;
    std::vector<Vec3> _forces;
    std::vector<Vec3> _sizes;
    BoundaryBox _bound;
    SpinLock _locker;
    std::atomic<bool> _active;
    FlatDynamicOctree _octree;
};
