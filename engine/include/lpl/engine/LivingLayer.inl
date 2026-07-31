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

inline core::u32 LivingLayer::bodiesFor(core::u32 species) const noexcept
{
    const core::u32 counted =
        _web.species.size() < _params.speciesCount ? static_cast<core::u32>(_web.species.size()) : _params.speciesCount;
    if (species >= counted)
        return 0u;

    math::Fixed32 total{};
    for (core::u32 i = 0u; i < counted; ++i)
        total = total + _web.species[i].population;

    const core::f32 totalF = total.toFloat();
    const core::f32 budget = static_cast<core::f32>(_params.maxBodies);
    const core::f32 scale = totalF > budget ? budget / totalF : 1.0f;
    const core::f32 wanted = _web.species[species].population.toFloat() * scale;
    return static_cast<core::u32>(wanted < 0.0f ? 0.0f : wanted);
}

} // namespace lpl::engine

#endif // LPL_ENGINE_LIVING_LAYER_INL
