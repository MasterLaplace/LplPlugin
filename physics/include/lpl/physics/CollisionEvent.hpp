/**
 * @file CollisionEvent.hpp
 * @brief Physics collision event definition for the EventBus.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PHYSICS_COLLISION_EVENT_HPP
#    define LPL_PHYSICS_COLLISION_EVENT_HPP

#    include <lpl/ecs/Entity.hpp>
#    include <lpl/math/Vec3.hpp>

namespace lpl::physics {

/**
 * @struct CollisionEvent
 * @brief Emitted by the PhysicsSystem when two entities collide.
 */
struct CollisionEvent {
    ecs::EntityId entityA;
    ecs::EntityId entityB;
    math::Vec3<float> normal;
    float impulseMagnitude;
};

} // namespace lpl::physics

#endif // LPL_PHYSICS_COLLISION_EVENT_HPP
