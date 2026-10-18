#ifndef LPL_ENGINE_SYSTEMS_CREATURESYSTEMS_INL
#define LPL_ENGINE_SYSTEMS_CREATURESYSTEMS_INL

namespace lpl::engine::systems {

template <typename Visit> void forEachCreature(const ecs::Registry &registry, Visit &&visit)
{
    for (const auto &partition : registry.partitions())
    {
        if (!partition || !partition->archetype().has(ecs::ComponentId::Position) ||
            !partition->archetype().has(ecs::ComponentId::Creature) ||
            !partition->archetype().has(ecs::ComponentId::Genome))
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            const auto *positions =
                static_cast<const math::Vec3<math::Fixed32> *>(chunk->latestComponent(ecs::ComponentId::Position));
            const auto *creature = static_cast<const core::u32 *>(chunk->latestComponent(ecs::ComponentId::Creature));
            const auto *genome = static_cast<const ecology::Genome *>(chunk->latestComponent(ecs::ComponentId::Genome));
            if (positions == nullptr || creature == nullptr || genome == nullptr)
                continue;
            for (core::u32 i = 0u; i < chunk->count(); ++i)
                visit(creature[i * 2u], creature[i * 2u + 1u], genome[i], positions[i]);
        }
    }
}

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_CREATURESYSTEMS_INL
