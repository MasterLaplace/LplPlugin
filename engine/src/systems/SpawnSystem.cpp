#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/engine/systems/SpawnSystem.hpp>

namespace lpl::engine::systems {

static const ecs::ComponentAccess kSpawnAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Velocity, ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::AABB,     ecs::AccessMode::ReadWrite},
    {ecs::ComponentId::Health,   ecs::AccessMode::ReadWrite},
};

static const ecs::SystemDescriptor kSpawnSystemDesc{"SpawnSystem", ecs::SchedulePhase::PrePhysics,
                                                    std::span<const ecs::ComponentAccess>{kSpawnAccesses}};

SpawnSystem::SpawnSystem(ecs::Registry &registry, const core::u32 &myEntityId, const bool &connected)
    : _registry(registry), _myEntityId(myEntityId), _connected(connected)
{
}

const ecs::SystemDescriptor &SpawnSystem::descriptor() const noexcept { return kSpawnSystemDesc; }

void SpawnSystem::execute(core::f32 /*dt*/)
{
    if (!_spawned && _connected && _myEntityId != ecs::EntityId::kNull &&
        !_registry.isAlive(ecs::EntityId{_myEntityId}))
    {
        ecs::Archetype arch;
        arch.add(ecs::ComponentId::Position);
        arch.add(ecs::ComponentId::Velocity);
        arch.add(ecs::ComponentId::AABB);
        arch.add(ecs::ComponentId::Health);

        // Create the avatar UNDER the server-assigned id (from the welcome), not a
        // fresh local one — otherwise CameraSystem/input, which name the entity by
        // _myEntityId, could never find it, and a reconciliation snapshot for the
        // same id would spawn a second copy. Idempotent with StateReconciliation:
        // whichever runs first, the other adopts the same live id.
        [[maybe_unused]] auto res = _registry.createEntityWithId(ecs::EntityId{_myEntityId}, arch);
        _spawned = true;
    }
}

} // namespace lpl::engine::systems
