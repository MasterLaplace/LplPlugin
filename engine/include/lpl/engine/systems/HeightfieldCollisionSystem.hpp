/**
 * @file HeightfieldCollisionSystem.hpp
 * @brief Keeping loose bodies on top of a bounded heightfield, and letting them slide.
 *
 * This lived in a viewer's `main.cpp` under the name `TerrainCollisionSystem`, and
 * its own comment argued that it belonged there: "a heightfield is content, not a
 * host service, so the collision against it belongs to the World". That argument is
 * wrong, and it is worth saying why, because it is the argument that keeps engine
 * knowledge trapped in apps.
 *
 * A heightfield *is* content. Colliding a rigid body against one is not. Nothing in
 * here knows which world it is standing on, what generated it, or what the bodies
 * are: it needs a grid of heights, a cell size, and Position/Velocity. Every bounded
 * heightfield world needs exactly this, and the one that had it could not be tested
 * because the code was in an executable.
 *
 * Three things it knows that are easy to lose, each learned from a picture:
 *  - **Zero mass means immovable.** Correcting an immovable body to the ground is
 *    harmless; handing it the downhill slide below is not, and it walks a tree off
 *    its own footing.
 *  - **Keep bodies over the map.** One nudged past the edge by a collision sails off
 *    and hangs in empty space, which looks exactly like the collision having failed.
 *  - **The slide is the point.** Without it a boulder stops where it landed and the
 *    terrain might as well be a floor; with it, the map's drainage pattern becomes
 *    visible in where things collect.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-04
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_HEIGHTFIELD_COLLISION_SYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_HEIGHTFIELD_COLLISION_SYSTEM_HPP

#    include <lpl/ecs/Registry.hpp>
#    include <lpl/ecs/System.hpp>
#    include <lpl/math/Vec3.hpp>
#    include <lpl/procgen/Heightfield.hpp>

namespace lpl::engine::systems {

/**
 * @class HeightfieldCollisionSystem
 * @brief Lands falling bodies on a heightfield, clamps them to it, slides them downhill.
 *
 * Registered in @c PostPhysics, deliberately. Doing this work in @c Physics would put
 * a second writer in the phase that already owns integration, and the engine's own
 * physics would be stepping the same buffers alongside it. Correcting positions
 * *after* the step is both the right order physically and the only order that keeps
 * one writer per phase.
 *
 * The field is held by reference: a world that regenerates its terrain in place keeps
 * working, and one that replaces the object must re-register.
 */
class HeightfieldCollisionSystem final : public ecs::ISystem {
public:
    /**
     * @param registry Where the bodies live.
     * @param terrain  Heights in cells; the grid is centred on the origin.
     * @param cellSize World units per cell.
     */
    HeightfieldCollisionSystem(ecs::Registry &registry, const procgen::Heightfield &terrain,
                              core::f32 cellSize) noexcept;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override { return _descriptor; }

    void execute(core::f32 dt) override;

    [[nodiscard]] core::u32 executions() const noexcept { return _executions; }

    /**
     * @brief Bodies touching the ground on the most recent tick.
     *
     * The honest measure of "at rest". Testing the vertical velocity instead reports
     * nothing ever settling: a body held up by position correction still has one tick
     * of gravity applied to it every tick, so its velocity oscillates around zero
     * forever even though it has not moved. What is stable is the contact.
     */
    [[nodiscard]] core::u32 resting() const noexcept { return _resting; }

    /// How hard a slope pushes a resting body along it, per tick.
    void setSlide(core::f32 slide) noexcept { _slide = slide; }
    /// What a resting body keeps of its horizontal speed each tick.
    void setFriction(core::f32 friction) noexcept { _friction = friction; }

private:
    static constexpr ecs::ComponentAccess kAccesses[] = {
        {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
        {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
        {ecs::ComponentId::AABB,     ecs::AccessMode::ReadOnly },
    };

    core::u32 _executions{0u};
    core::u32 _resting{0u};
    ecs::SystemDescriptor _descriptor{"HeightfieldCollision", ecs::SchedulePhase::PostPhysics, kAccesses};
    ecs::Registry &_registry;
    const procgen::Heightfield &_terrain;
    core::f32 _cellSize;
    core::f32 _slide{0.55f};
    core::f32 _friction{0.86f};
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_HEIGHTFIELD_COLLISION_SYSTEM_HPP
