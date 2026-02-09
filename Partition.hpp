#pragma once

#include "Math.hpp"
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
    };

    struct EntitySnapshot {
        uint64_t id;
        Vec3 position;
        Quat rotation;
        Vec3 velocity;
        float mass;
    };

public:
    Partition(Vec3 position, float size) noexcept : _bound({
        {position.x, std::numeric_limits<float>::lowest(), position.z},
        {position.x + size, std::numeric_limits<float>::max(), position.z + size}
    }), _active(true) {}
    Partition(Partition &&other) noexcept :
        _ids(std::move(other._ids)),
        _positions(std::move(other._positions)),
        _rotations(std::move(other._rotations)),
        _velocities(std::move(other._velocities)),
        _masses(std::move(other._masses)),
        _bound(std::move(other._bound)),
        _active(other._active.load()) {}

    Partition &operator=(Partition &&other) noexcept {
        _ids = std::move(other._ids);
        _positions = std::move(other._positions);
        _rotations = std::move(other._rotations);
        _velocities = std::move(other._velocities);
        _masses = std::move(other._masses);
        _bound = std::move(other._bound);
        _active = other._active.load();
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
    }

    [[nodiscard]] EntityRef getEntity(const size_t index) noexcept {return EntityRef{
        _positions[index],
        _rotations[index],
        _velocities[index],
        _masses[index]
    }; }

    void physicsTick(float deltatime) noexcept
    {
        LocalGuard guard(_locker);

        for (uint64_t index = 0u; index < _ids.size(); ++index)
        {
            _positions[index] += _velocities[index] * deltatime;
            if (!_bound.contains(_positions[index]))
                continue;

            EntitySnapshot moved_entity;
            moved_entity.id = _ids[index];
            moved_entity.position = _positions[index];
            moved_entity.id = _ids[index];
            moved_entity.id = _ids[index];
            moved_entity.id = _ids[index];
        }
    }

private:
    std::vector<uint64_t> _ids;
    std::vector<Vec3> _positions;
    std::vector<Quat> _rotations;
    std::vector<Vec3> _velocities;
    std::vector<float> _masses;
    BoundaryBox _bound;
    SpinLock _locker;
    std::atomic<bool> _active;
};
