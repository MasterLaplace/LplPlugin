#ifndef LPL_ECOLOGY_HERD_INL
#define LPL_ECOLOGY_HERD_INL

namespace lpl::ecology {

inline SpeciesScent defaultGrazerScent() noexcept
{
    SpeciesScent scent;
    scent.depositChannel = static_cast<core::u32>(ai::ScentChannel::Herbivore);
    scent.palate.add(ai::ScentChannel::Plant, math::Fixed32::one());
    scent.palate.add(ai::ScentChannel::Carnivore, -math::Fixed32::fromInt(3));
    scent.palate.add(ai::ScentChannel::Terror, -math::Fixed32::fromInt(4));
    scent.palate.add(ai::ScentChannel::Herbivore, -math::Fixed32::fromFloat(0.25f));
    return scent;
}

inline SpeciesScent defaultHunterScent() noexcept
{
    SpeciesScent scent;
    scent.depositChannel = static_cast<core::u32>(ai::ScentChannel::Carnivore);
    scent.palate.add(ai::ScentChannel::Herbivore, math::Fixed32::one());
    scent.palate.add(ai::ScentChannel::Carnivore, -math::Fixed32::fromFloat(0.6f));
    scent.palate.add(ai::ScentChannel::Terror, -math::Fixed32::fromInt(2));
    return scent;
}

inline void applyDefaultScents(HerdParams &params) noexcept
{
    params.scent[0] = defaultGrazerScent();
    params.scent[1] = defaultHunterScent();
}

inline const core::u32 *Herd::creatureOf(ecs::Registry &registry, ecs::EntityId entity) noexcept
{
    core::u32 row = 0u;
    ecs::Chunk *chunk = registry.chunkOf(entity, row);
    if (chunk == nullptr || !chunk->archetype().has(ecs::ComponentId::Creature))
        return nullptr;
    const auto *creature = static_cast<const core::u32 *>(chunk->latestComponent(ecs::ComponentId::Creature));
    if (creature == nullptr)
        return nullptr;
    return creature + static_cast<core::usize>(row) * 2u;
}

inline void Herd::clear()
{
    if (_registry != nullptr)
        for (core::usize i = 0u; i < _bodies.size(); ++i)
            (void) _registry->destroyEntity(_bodies[i]);
    _bodies.clear();
}

inline core::u32 Herd::countSpecies(core::u32 species) const noexcept
{
    if (_registry == nullptr)
        return 0u;
    core::u32 count = 0u;
    for (core::usize i = 0u; i < _bodies.size(); ++i)
    {
        const core::u32 *creature = creatureOf(*_registry, _bodies[i]);
        count += creature != nullptr && creature[0] == species ? 1u : 0u;
    }
    return count;
}

inline bool Herd::removeOne(core::u32 species) noexcept
{
    if (_registry == nullptr)
        return false;
    for (core::usize i = 0u; i < _bodies.size(); ++i)
    {
        const core::u32 *creature = creatureOf(*_registry, _bodies[i]);
        if (creature == nullptr || creature[0] != species)
            continue;
        (void) _registry->destroyEntity(_bodies[i]);
        _bodies[i] = _bodies[_bodies.size() - 1u];
        _bodies.pop_back();
        return true;
    }
    return false;
}

template <typename CountFor, typename Spawn>
inline void Herd::reconcile(core::u32 speciesCount, CountFor &&countFor, Spawn &&spawn)
{
    for (core::u32 species = 0u; species < speciesCount; ++species)
    {
        const core::u32 wanted = countFor(species);
        core::u32 present = countSpecies(species);
        while (present < wanted && spawn(species))
            ++present;
        while (present > wanted && removeOne(species))
            --present;
    }
}

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_HERD_INL
