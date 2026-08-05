/**
 * @file LivingLayer.inl
 * @brief Out-of-line definitions for engine::LivingLayer.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_LIVING_LAYER_INL
#    define LPL_ENGINE_LIVING_LAYER_INL

namespace lpl::engine {

inline void LivingLayer::configure(const LivingLayerParams &params, const ecology::LivingRecipe &recipe, core::u32 seed)
{
    _params = params;
    _recipe = recipe;
    _seed = seed;
    _heredity = seed ^ 0xA57E22u;
    _nextId = 1u;
    _grazed = 0u;
    _herd.clear();
    // The one HerdParams this layer runs on: the systems hold a reference to it,
    // so it is configured here rather than rebuilt each step.
    _herdParams.speciesCount = params.speciesCount;
    ecology::applyDefaultScents(_herdParams);
}

inline void LivingLayer::seedWeb(core::u32 standingPlants)
{
    // Zero means "the world cannot count its plants yet" — a streamed world has
    // no total, only what is resident — and then the recipe's own capacity
    // stands. Overriding it with zero starves the producers on the first tick,
    // and a food web whose bottom is empty inverts: three grazers, forty-four
    // hunters, which is what the screen showed.
    _web = ecology::TrophicWeb{};
    const core::u32 declared =
        _recipe.speciesCount < ecology::kMaxLivingSpecies ? _recipe.speciesCount : ecology::kMaxLivingSpecies;
    for (core::u32 i = 0u; i < declared; ++i)
    {
        ecology::SpeciesParams params = _recipe.species[i].params;
        math::Fixed32 initial = _recipe.species[i].initial;
        if (standingPlants != 0u && params.level == ecology::TrophicLevel::Producer)
        {
            params.capacity = math::Fixed32::fromInt(static_cast<core::i32>(standingPlants + 1u));
            initial = params.capacity;
        }
        (void) _web.add(params, initial, _recipe.species[i].preyIndex);
    }
}

inline core::u32 LivingLayer::webIndexOfBodied(core::u32 bodied) const noexcept
{
    core::u32 seen = 0u;
    for (core::u32 i = 0u; i < _web.species.size(); ++i)
    {
        if (_web.species[i].params.level == ecology::TrophicLevel::Producer)
            continue;
        if (seen == bodied)
            return i;
        ++seen;
    }
    return kNoSpecies;
}

inline core::u32 LivingLayer::rawBodiesFor(core::u32 webIndex) const noexcept
{
    if (webIndex >= _web.species.size())
        return 0u;
    const ecology::Species &species = _web.species[webIndex];
    // At or below its refuge the species is gone as far as the model is concerned,
    // and drawing a body for it would contradict the census.
    if (species.population <= species.params.refuge)
        return 0u;

    const core::u32 perBody = _recipe.headPerBody != 0u ? _recipe.headPerBody : 2u;
    core::i32 wanted = species.population.toInt() / static_cast<core::i32>(perBody);
    // Above its refuge, so it exists — and a world where existing is invisible is
    // not showing the model. This is the floor that was learned the hard way: at one
    // body per ten head, a population of seven floored to zero and the map stayed
    // empty while the HUD cheerfully reported a working ecosystem. The spawning, the
    // flocking and the drawing were all correct and all ran on an empty list.
    if (wanted < 1)
        wanted = 1;
    return static_cast<core::u32>(wanted);
}

inline core::u32 LivingLayer::bodiesFor(core::u32 species) const noexcept
{
    if (species >= _params.speciesCount)
        return 0u;
    const core::u32 index = webIndexOfBodied(species);
    if (index == kNoSpecies)
        return 0u;

    const core::u32 mine = rawBodiesFor(index);
    core::u32 total = 0u;
    for (core::u32 i = 0u; i < _params.speciesCount; ++i)
    {
        const core::u32 other = webIndexOfBodied(i);
        if (other != kNoSpecies)
            total += rawBodiesFor(other);
    }
    if (total <= _params.maxBodies || total == 0u)
        return mine;

    // Proportional, so the shape of the web survives the ceiling.
    const core::u32 scaled = (mine * _params.maxBodies) / total;
    return scaled == 0u && mine != 0u ? 1u : scaled;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_LIVING_LAYER_INL
