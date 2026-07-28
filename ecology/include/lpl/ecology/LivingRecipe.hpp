/**
 * @file LivingRecipe.hpp
 * @brief A replayable *running simulation*: the second half of the parity gate.
 *
 * `procgen::WorldRecipe` puts the world's **shape** under the determinism
 * contract — the height field, the biome map, where the entities were placed. It
 * cannot say anything about what happens next, because it stops the moment the
 * world exists. So `ai/` and `ecology/` shipped linked into ring 0 with no
 * contract at all: their determinism was checked on the host and assumed on the
 * target, which is exactly the assumption this project refuses to make anywhere
 * else.
 *
 * A living recipe is the same idea one step later: a seed, a fixed number of
 * ticks, and the four subsystems whose state evolves — a trophic web, a breeding
 * population, a pheromone field with agents walking it, and an abstract world
 * migrating creatures between rooms. Running it on the Linux oracle and inside
 * the kernel must fold the SAME signatures, bit for bit.
 *
 * The four signatures are separate on purpose, and for the reason the world gate
 * folds three grids rather than one: a single number tells you that something
 * diverged, and nothing about where. Populations, genomes, the field and the
 * social layer fail for entirely different reasons, and a gate that cannot say
 * which one moved is a gate you have to debug by bisection.
 *
 * @warning Everything here is authoritative Fixed32. The `core::f32` knobs in the
 *          parameter structs are converted once through `Fixed32::fromFloat` at
 *          the top of each pass — the same discipline `procgen` follows — so no
 *          float arithmetic ever reaches a value that gets folded.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECOLOGY_LIVINGRECIPE_HPP
#    define LPL_ECOLOGY_LIVINGRECIPE_HPP

#    include <lpl/ai/AbstractWorld.hpp>
#    include <lpl/ai/StigmergyField.hpp>
#    include <lpl/ai/Swarm.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecology/Genome.hpp>
#    include <lpl/ecology/Populations.hpp>
#    include <lpl/ecology/Society.hpp>

namespace lpl::ecology {

/**
 * @struct LivingRecipe
 * @brief The declarative description of a deterministic simulation run.
 *
 * Bounded everywhere, for the same reason `WorldRecipe` is: this is a wire object
 * before it is an engine object, and ring 0 has neither a heap to grow into nor a
 * reason to trust a length field it did not write.
 */
/// Species a single living recipe may declare. Bounded for the same reason the
/// scatter rules are: this is a wire object before it is an engine object.
inline constexpr core::u32 kMaxLivingSpecies = 4u;

/**
 * @struct LivingSpecies
 * @brief One authored population: its demography and what it eats.
 */
struct LivingSpecies {
    SpeciesParams params{};                ///< Growth, mortality, predation, capacity, refuge.
    math::Fixed32 initial{};               ///< Head count at tick 0.
    core::u32 preyIndex{Species::kNoPrey}; ///< Index into the table; kNoPrey for a producer.
};

struct LivingRecipe {
    core::u32 seed{2027u}; ///< Master seed; every subsystem derives its own stream.
    core::u32 ticks{48u};  ///< Steps to run. The fold is taken after the last one.
    /// Duration of one step, in seconds. Explicit because anything integrating a
    /// velocity needs it, and an implicit 1 is how a flock ends up at mach 20.
    math::Fixed32 stepSeconds{math::Fixed32::fromRaw(1092)}; // 1/60 s

    core::u32 width{24u};   ///< Stigmergy field columns.
    core::u32 depth{24u};   ///< Stigmergy field rows.
    core::u32 channels{2u}; ///< Stigmergy channels: a trail and a scent.

    core::u32 rooms{12u};       ///< Rooms in the abstract world.
    core::u32 creatures{24u};   ///< Abstract creatures migrating between them.
    core::u32 ants{8u};         ///< Agents walking the pheromone field.
    core::u32 boids{16u};       ///< Flocking bodies.
    core::u32 genomes{16u};     ///< Breeding population size (kept constant).
    core::u32 packMembers{16u}; ///< Animals in the social layer.

    /**
     * @brief The food web, as data.
     *
     * It used to be four species hardcoded inside `runLiving`, which made the
     * gate's ecosystem unauthorable: a `.lplscene` could describe the terrain
     * down to the erosion iteration count and had no way to say what lived on
     * it. @ref parityLivingRecipe declares exactly the web the hardcoded version
     * built, so the folds are unchanged by the move.
     */
    LivingSpecies species[kMaxLivingSpecies]{};
    core::u32 speciesCount{0u};

    /// Ticks a grazed plant takes to come back, for hosts that model vegetation.
    core::u32 regrowthTicks{900u};
    /// Head count one drawn body stands for; 0 means the host decides.
    core::u32 headPerBody{2u};

    ai::StigmergyParams stigmergy{}; ///< Evaporation, diffusion, floor.
    ai::AntParams foraging{};        ///< Exploration balance.
    ai::BoidParams flock{};          ///< Separation, alignment, cohesion.
    ai::RealizationBudget budget{};  ///< How many rooms may hold bodies.
    HeredityParams heredity{};       ///< Mutation, meltdown, anomaly threshold.
    PackParams packs{};              ///< Pack life-cycle thresholds.
};

/**
 * @struct LivingResult
 * @brief What running a living recipe produced.
 *
 * Free of Fixed32 and bool, like @ref procgen::WorldRecipeResult, so a C caller
 * can copy it field by field with no conversion that could differ between
 * targets.
 */
struct LivingResult {
    core::u32 populationSignature{0u}; ///< FNV-1a fold of every species' head count.
    core::u32 genomeSignature{0u};     ///< FNV-1a fold of every gene of every genome.
    core::u32 stigmergySignature{0u};  ///< FNV-1a fold of every channel of the field.
    core::u32 socialSignature{0u};     ///< FNV-1a fold of the abstract world, the flock and the packs.

    core::u32 extinctions{0u};   ///< Species that fell to their refuge floor.
    core::u32 anomalies{0u};     ///< Genomes standing k sigma above the species mean.
    core::u32 realisedRooms{0u}; ///< Rooms holding bodies at the end.
    core::u32 migrations{0u};    ///< Abstract room transitions over the whole run.
    core::u32 alphaChanges{0u};  ///< Times a pack changed leader.
    core::u32 trailCells{0u};    ///< Field cells still above the evaporation floor.
    core::u32 ok{0u};            ///< 1 when the run is well formed (see below).
};

/**
 * @brief The canonical recipe run by both the Linux oracle and the kernel.
 *
 * Sized like @ref procgen::parityWorldRecipe: a 24x24 field, a few dozen agents,
 * forty-eight ticks. Enough that every subsystem's state actually moves — a gate
 * over a simulation that has not evolved is a gate over its initial conditions —
 * and small enough to run inside the boot battery on a 4 MiB heap.
 *
 * Changing anything here re-folds the gate on BOTH sides at once, which is the
 * point of it living in one constexpr function.
 *
 * @return The parity living recipe.
 */
[[nodiscard]] constexpr LivingRecipe parityLivingRecipe() noexcept
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

    // A field that forgets fast is the interesting one to fold: with a slow
    // evaporation every cell saturates within a few ticks and the signature
    // stops depending on where the agents actually walked.
    recipe.stigmergy.evaporation = 0.88f;
    recipe.stigmergy.diffusion = 0.10f;

    recipe.budget.maxRealisedRooms = 4u;

    // Above the default, deliberately: at one chance in sixteen a sixteen-genome
    // population over forty-eight generations barely mutates, and a gate should
    // exercise the path it exists to protect.
    recipe.heredity.mutationChance16 = 3u;

    // The four-level web the run exercises: each level eats the one below, so the
    // cascade the module exists to produce is actually in the fold — remove the
    // top and the mesopredator is released. Written out here rather than built in
    // the run, so a cartridge can describe its own.
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

/**
 * @brief Runs @p recipe to completion and folds every subsystem.
 *
 * The tick order is part of the contract and fixed here, not by the caller:
 * populations, then heredity, then the field and its agents, then the flock, then
 * the abstract world, then the social layer. Reordering changes which random
 * stream advances when, and therefore the folds.
 *
 * @param recipe What to run.
 * @return The four signatures and the counters behind them.
 */
LivingResult runLiving(const LivingRecipe &recipe);

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_LIVINGRECIPE_HPP
