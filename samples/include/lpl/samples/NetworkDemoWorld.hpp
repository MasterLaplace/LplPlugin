/**
 * @file NetworkDemoWorld.hpp
 * @brief The engine's networked demo content as a World.
 *
 * This is the 50-NPC scene that used to be hard-coded inside Engine::init. It is
 * game CONTENT, not engine machinery, so it lives here as a World a server
 * injects — the engine itself now knows about no particular game.
 *
 * The generic systems this world relies on (physics, networking) are engine
 * built-ins selected through Config::Builder (enablePhysics / enableNetworking);
 * this World only provides the entities.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-21
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SAMPLES_NETWORKDEMOWORLD_HPP
#    define LPL_SAMPLES_NETWORKDEMOWORLD_HPP

#    include <lpl/core/Log.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/ecs/Component.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/engine/World.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/math/Vec3.hpp>

namespace lpl::samples {

/**
 * @class NetworkDemoWorld
 * @brief Deterministic 50-NPC world used by the networked server demo.
 */
class NetworkDemoWorld final : public engine::World {
public:
    /**
     * @brief Spawns the NPCs.
     *
     * Placement comes from a simple deterministic LCG (seed 42), so every run spawns the same NPCs
     * at the same places: x and z in [-100, 100], y in [0, 50]. Position is written to both the
     * front and the back buffer, as authoritative Fixed32.
     */
    [[nodiscard]] inline core::Expected<void> onInit(engine::WorldContext &context) override;

    [[nodiscard]] const char *name() const noexcept override { return "NetworkDemo"; }
};

} // namespace lpl::samples

#    include "NetworkDemoWorld.inl"

#endif // LPL_SAMPLES_NETWORKDEMOWORLD_HPP
