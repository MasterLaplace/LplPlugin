#ifndef LPL_SAMPLES_NETWORKDEMOWORLD_INL
#define LPL_SAMPLES_NETWORKDEMOWORLD_INL

namespace lpl::samples {

inline core::Expected<void> NetworkDemoWorld::onInit(engine::WorldContext &context)
{
    (void) context;
    core::Log::info("NetworkDemoWorld: spawning NPC entities");

    ecs::Archetype npcArch;
    npcArch.add(ecs::ComponentId::Position);
    npcArch.add(ecs::ComponentId::Velocity);
    npcArch.add(ecs::ComponentId::Mass);
    npcArch.add(ecs::ComponentId::AABB);
    npcArch.add(ecs::ComponentId::Health);

    core::u32 seed = 42;
    auto nextRand = [&seed]() -> float {
        seed = seed * 1103515245u + 12345u;
        return static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
    };

    static constexpr core::u32 kNpcCount = 50;
    for (core::u32 i = 0; i < kNpcCount; ++i)
    {
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

        float px = (nextRand() - 0.5f) * 200.0f;
        float py = nextRand() * 50.0f;
        float pz = (nextRand() - 0.5f) * 200.0f;

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

        math::Vec3<math::Fixed32> vel{math::Fixed32::zero(), math::Fixed32::zero(), math::Fixed32::zero()};
        if (auto *wvel = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Velocity)))
        {
            wvel[ref.localIndex] = vel;
        }

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

        math::Vec3<math::Fixed32> size{math::Fixed32::fromInt(1), math::Fixed32::fromInt(2), math::Fixed32::fromInt(1)};
        if (auto *wsize = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
        {
            wsize[ref.localIndex] = size;
        }
        if (auto *rsize = const_cast<math::Vec3<math::Fixed32> *>(
                static_cast<const math::Vec3<math::Fixed32> *>(chunk.readComponent(ecs::ComponentId::AABB))))
        {
            rsize[ref.localIndex] = size;
        }

        core::i32 hp = 100;
        if (auto *whp = static_cast<core::i32 *>(chunk.writeComponent(ecs::ComponentId::Health)))
        {
            whp[ref.localIndex] = hp;
        }
        if (auto *rhp =
                const_cast<core::i32 *>(static_cast<const core::i32 *>(chunk.readComponent(ecs::ComponentId::Health))))
        {
            rhp[ref.localIndex] = hp;
        }

        auto fixedPos = math::Vec3<math::Fixed32>{math::Fixed32::fromFloat(px), math::Fixed32::fromFloat(py),
                                                  math::Fixed32::fromFloat(pz)};
        [[maybe_unused]] auto res = spatialPartition()->insertOrUpdate(entityId, fixedPos);
    }

    return {};
}

} // namespace lpl::samples

#endif // LPL_SAMPLES_NETWORKDEMOWORLD_INL
