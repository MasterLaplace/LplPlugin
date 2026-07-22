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
    [[nodiscard]] core::Expected<void> onInit(engine::WorldContext &context) override
    {
        (void) context;
        core::Log::info("NetworkDemoWorld: spawning NPC entities");

        // Build the NPC archetype: Position, Velocity, Mass, AABB, Health
        ecs::Archetype npcArch;
        npcArch.add(ecs::ComponentId::Position);
        npcArch.add(ecs::ComponentId::Velocity);
        npcArch.add(ecs::ComponentId::Mass);
        npcArch.add(ecs::ComponentId::AABB);
        npcArch.add(ecs::ComponentId::Health);

        // Simple deterministic LCG (seed 42) for reproducible NPC placement.
        // IDs 0–99 are reserved for NPCs; player entities start at 100.
        core::u32 seed = 42;
        auto nextRand = [&seed]() -> float {
            seed = seed * 1103515245u + 12345u;
            return static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
        };

        static constexpr core::u32 kNpcCount = 50;
        for (core::u32 i = 0; i < kNpcCount; ++i)
        {
            // Create entity in Registry with proper SoA component storage
            auto entityResult = registry().createEntity(npcArch);
            if (!entityResult.has_value())
                continue;

            auto entityId = entityResult.value();
            auto refResult = registry().resolve(entityId);
            if (!refResult.has_value())
                continue;

            auto ref = refResult.value();
            auto &partition = registry().getOrCreatePartition(npcArch);
            const auto &chunks = partition.chunks();
            if (ref.chunkIndex >= static_cast<core::u32>(chunks.size()))
                continue;

            auto &chunk = *chunks[ref.chunkIndex];

            float px = (nextRand() - 0.5f) * 200.0f; // [-100, 100]
            float py = nextRand() * 50.0f;           // [0, 50]
            float pz = (nextRand() - 0.5f) * 200.0f; // [-100, 100]

            // Write position to both front and back buffers (authoritative Fixed32)
            math::Vec3<math::Fixed32> pos{math::Fixed32::fromFloat(px), math::Fixed32::fromFloat(py),
                                          math::Fixed32::fromFloat(pz)};
            if (auto *wpos = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position)))
            {
                wpos[ref.localIndex] = pos;
            }
            if (auto *rpos = const_cast<math::Vec3<math::Fixed32> *>(
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk.readComponent(ecs::ComponentId::Position))))
            {
                rpos[ref.localIndex] = pos;
            }

            // Write velocity (zero initially)
            math::Vec3<math::Fixed32> vel{math::Fixed32::zero(), math::Fixed32::zero(), math::Fixed32::zero()};
            if (auto *wvel = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Velocity)))
            {
                wvel[ref.localIndex] = vel;
            }

            // Write mass
            math::Fixed32 mass = math::Fixed32::one();
            if (auto *wmass = static_cast<math::Fixed32 *>(chunk.writeComponent(ecs::ComponentId::Mass)))
            {
                wmass[ref.localIndex] = mass;
            }
            if (auto *rmass = const_cast<math::Fixed32 *>(
                    static_cast<const math::Fixed32 *>(chunk.readComponent(ecs::ComponentId::Mass))))
            {
                rmass[ref.localIndex] = mass;
            }

            // Write AABB (size)
            math::Vec3<math::Fixed32> size{math::Fixed32::fromInt(1), math::Fixed32::fromInt(2),
                                           math::Fixed32::fromInt(1)};
            if (auto *wsize = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
            {
                wsize[ref.localIndex] = size;
            }
            if (auto *rsize = const_cast<math::Vec3<math::Fixed32> *>(
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk.readComponent(ecs::ComponentId::AABB))))
            {
                rsize[ref.localIndex] = size;
            }

            // Write health
            core::i32 hp = 100;
            if (auto *whp = static_cast<core::i32 *>(chunk.writeComponent(ecs::ComponentId::Health)))
            {
                whp[ref.localIndex] = hp;
            }
            if (auto *rhp = const_cast<core::i32 *>(
                    static_cast<const core::i32 *>(chunk.readComponent(ecs::ComponentId::Health))))
            {
                rhp[ref.localIndex] = hp;
            }

            // Update spatial index
            auto fixedPos = math::Vec3<math::Fixed32>{math::Fixed32::fromFloat(px), math::Fixed32::fromFloat(py),
                                                      math::Fixed32::fromFloat(pz)};
            [[maybe_unused]] auto res = spatialPartition()->insertOrUpdate(entityId, fixedPos);
        }

        return {};
    }

    [[nodiscard]] const char *name() const noexcept override { return "NetworkDemo"; }
};

} // namespace lpl::samples

#endif // LPL_SAMPLES_NETWORKDEMOWORLD_HPP
