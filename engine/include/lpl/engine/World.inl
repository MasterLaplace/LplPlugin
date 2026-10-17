#ifndef LPL_ENGINE_WORLD_INL
#define LPL_ENGINE_WORLD_INL

namespace lpl::engine {

inline core::u64 World::stateHash() const noexcept
{
    core::u64 accumulator = 0;

    for (const auto &partition : _registry.partitions())
    {
        if (!partition)
            continue;

        const auto &archetype = partition->archetype();
        const bool hasPosition = archetype.has(ecs::ComponentId::Position);
        const bool hasVelocity = archetype.has(ecs::ComponentId::Velocity);
        if (!hasPosition && !hasVelocity)
            continue;

        for (const auto &chunk : partition->chunks())
        {
            const core::u32 count = chunk->count();
            if (count == 0)
                continue;

            const auto *positions =
                hasPosition ?
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk->latestComponent(ecs::ComponentId::Position)) :
                    nullptr;
            const auto *velocities =
                hasVelocity ?
                    static_cast<const math::Vec3<math::Fixed32> *>(chunk->latestComponent(ecs::ComponentId::Velocity)) :
                    nullptr;

            const auto entityIds = chunk->entities();

            for (core::u32 i = 0; i < count; ++i)
            {
                math::StateHash entityHash;
                entityHash.combine(entityIds[i].raw());

                if (positions != nullptr)
                {
                    entityHash.combine(positions[i].x.raw());
                    entityHash.combine(positions[i].y.raw());
                    entityHash.combine(positions[i].z.raw());
                }
                if (velocities != nullptr)
                {
                    entityHash.combine(velocities[i].x.raw());
                    entityHash.combine(velocities[i].y.raw());
                    entityHash.combine(velocities[i].z.raw());
                }

                accumulator += entityHash.digest();
            }
        }
    }

    return accumulator;
}

inline ecs::WorldPartition &World::enableSpatialPartition(math::Fixed32 cellSize, core::u32 cellCapacity)
{
    if (!_spatial)
        _spatial = lpl::pmr::make_unique<ecs::WorldPartition>(cellSize, cellCapacity);
    return *_spatial;
}

inline LivingLayer &World::enableLivingLayer(const LivingLayerParams &params, const ecology::LivingRecipe &recipe,
                                             core::u32 seed)
{
    if (!_living)
    {
        _living = lpl::pmr::make_unique<LivingLayer>();
        _living->configure(params, recipe, seed);
    }
    return *_living;
}

inline core::Expected<void> World::onInit(WorldContext &context)
{
    (void) context;
    return {};
}

inline void World::onRender(WorldContext &context, core::f64 alpha)
{
    (void) context;
    (void) alpha;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_WORLD_INL
