/**
 * @file Herd.inl
 * @brief Out-of-line definitions for ecology::Herd.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-08-04
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECOLOGY_HERD_INL
#    define LPL_ECOLOGY_HERD_INL

namespace lpl::ecology {

inline SpeciesScent defaultGrazerScent() noexcept
{
    SpeciesScent scent;
    scent.depositChannel = static_cast<core::u32>(ai::ScentChannel::Herbivore);
    // Drawn to what it eats, driven off by what eats it, and mildly off its own
    // kind — the last term is what spreads a herd over a meadow instead of
    // letting it collapse into one cell of grass.
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
    // Attraction to the prey plus repulsion from other hunters. Those two terms
    // are the whole of the encirclement: pack members that all wanted the same
    // cell now want DIFFERENT cells around the same prey, and the flanking falls
    // out of the arithmetic rather than out of a tactic anybody wrote.
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
    // ecs::Registry::chunkOf does the walk AND the identity check: a chunk index
    // alone does not identify a chunk, since every partition has one, and skipping
    // that check reads another archetype's row. The walk was written out by hand here
    // and in a map viewer before the registry could answer it.
    core::u32 row = 0u;
    ecs::Chunk *chunk = registry.chunkOf(entity, row);
    if (chunk == nullptr || !chunk->archetype().has(ecs::ComponentId::Creature))
        return nullptr;
    // The WRITE side, like every other creature read: that is where the systems write.
    const auto *creature = static_cast<const core::u32 *>(chunk->writeComponent(ecs::ComponentId::Creature));
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
        // Destroy the ENTITY, not just the roster entry. Popping the entry alone
        // left the body in the registry: a species could go extinct in the census
        // while its animals kept being drawn and kept obstructing the living.
        (void) _registry->destroyEntity(_bodies[i]);
        _bodies[i] = _bodies[_bodies.size() - 1u];
        _bodies.pop_back();
        return true;
    }
    return false;
}

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_HERD_INL
