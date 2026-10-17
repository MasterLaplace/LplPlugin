#ifndef LPL_ECOLOGY_LIVINGRECIPE_INL
#define LPL_ECOLOGY_LIVINGRECIPE_INL

namespace lpl::ecology {

constexpr LivingRecipe parityLivingRecipe() noexcept
{
    LivingRecipe recipe{};

    recipe.seed = 2027u;
    recipe.ticks = 48u;
    recipe.width = 24u;
    recipe.depth = 24u;
    recipe.channels = 2u;
    recipe.rooms = 12u;
    recipe.creatures = 24u;
    recipe.ants = 8u;
    recipe.boids = 16u;
    recipe.genomes = 16u;
    recipe.packMembers = 16u;

    recipe.stigmergy.evaporation = 0.88f;
    recipe.stigmergy.diffusion = 0.10f;

    recipe.budget.maxRealisedRooms = 4u;

    recipe.heredity.mutationChance16 = 3u;

    recipe.species[0].params.level = TrophicLevel::Producer;
    recipe.species[0].params.capacity = math::Fixed32::fromInt(1000);
    recipe.species[0].initial = math::Fixed32::fromInt(800);
    recipe.species[0].preyIndex = Species::kNoPrey;

    recipe.species[1].params.level = TrophicLevel::Primary;
    recipe.species[1].params.capacity = math::Fixed32::fromInt(200);
    recipe.species[1].initial = math::Fixed32::fromInt(120);
    recipe.species[1].preyIndex = 0u;

    recipe.species[2].params.level = TrophicLevel::Secondary;
    recipe.species[2].params.capacity = math::Fixed32::fromInt(40);
    recipe.species[2].initial = math::Fixed32::fromInt(24);
    recipe.species[2].preyIndex = 1u;

    recipe.species[3].params.level = TrophicLevel::Apex;
    recipe.species[3].params.capacity = math::Fixed32::fromInt(8);
    recipe.species[3].initial = math::Fixed32::fromInt(5);
    recipe.species[3].preyIndex = 2u;

    recipe.speciesCount = 4u;

    return recipe;
}

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_LIVINGRECIPE_INL
