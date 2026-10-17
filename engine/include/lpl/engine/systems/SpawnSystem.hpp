#pragma once

#ifndef LPL_ENGINE_SYSTEMS_SPAWN_SYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_SPAWN_SYSTEM_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/System.hpp>

namespace lpl::ecs {
class Registry;
}

namespace lpl::engine::systems {

/**
 * @class SpawnSystem
 * @brief Spawns the local player entity on the client if connected and not yet registered.
 */
class SpawnSystem final : public ecs::ISystem {
public:
    SpawnSystem(ecs::Registry &registry, const core::u32 &myEntityId, const bool &connected);

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;

    /**
     * @brief Creates the local avatar once the welcome has named it.
     *
     * The avatar is created UNDER the server-assigned id (from the welcome), not a fresh
     * local one — otherwise CameraSystem/input, which name the entity by its id, could
     * never find it, and a reconciliation snapshot for the same id would spawn a second
     * copy. Idempotent with StateReconciliation: whichever runs first, the other adopts
     * the same live id.
     */
    void execute(core::f32 dt) override;

private:
    ecs::Registry &_registry;
    const core::u32 &_myEntityId;
    const bool &_connected;
    bool _spawned = false;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_SPAWN_SYSTEM_HPP
