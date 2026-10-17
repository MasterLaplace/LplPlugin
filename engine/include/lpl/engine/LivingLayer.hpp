/**
 * @file LivingLayer.hpp
 * @brief The living half of a world: a food web, a scent field, and bodies.
 *
 * Composes the three pieces that only mean something together — ecology::Herd walks,
 * ai::ScentWindow is what it walks by, ecology::TrophicWeb is what says how many of
 * it there should be — and owns the one rule that keeps them consistent: the bodies
 * are a SAMPLE of the population, scaled to a budget, preserving the ratio between
 * species.
 *
 * That rule is here because getting it wrong is invisible in the numbers and obvious
 * on screen: clamping each species at the same ceiling independently gave
 * twenty-four grazers and twenty-four hunters out of a web that says five to one,
 * which reads as a plague of predators.
 *
 * A world OWNS one of these. It is not a base class, and it knows nothing about
 * terrain: where an animal may stand and what it eats are asked of
 * engine::ITerrainQuery, by the systems, not by this layer.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_LIVING_LAYER_HPP
#    define LPL_ENGINE_LIVING_LAYER_HPP

#    include <lpl/ai/ScentWindow.hpp>
#    include <lpl/ecology/Genome.hpp>
#    include <lpl/ecology/Herd.hpp>
#    include <lpl/ecology/LivingRecipe.hpp>
#    include <lpl/ecology/Populations.hpp>
#    include <lpl/ecology/Vegetation.hpp>
#    include <lpl/ecs/Archetype.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/engine/systems/CreatureSystems.hpp>
#    include <lpl/math/Random.hpp>

namespace lpl::engine {

/**
 * @struct LivingLayerParams
 * @brief The budgets a host can afford for the living layer.
 */
struct LivingLayerParams {
    core::u32 maxBodies{48u};   /**< Bodies on screen, all species together. */
    core::u32 speciesCount{2u}; /**< Species that get bodies (grazer, hunter). */
    core::u32 scentSpan{64u};   /**< Cells across the pheromone window. */
    /**
     * Stigmergy channels. Six, because that is how many @c ai::ScentChannel names
     * and the herd now reads them by meaning: plant, herbivore, carnivore, terror,
     * kin, pheromone. Two was enough while every animal climbed one hard-coded
     * channel — which is exactly the bug that made a grazer follow its own scent.
     */
    core::u32 scentLayers{6u};
    core::u32 webPeriod{60u};     /**< Ticks between two food-web integrations. */
    core::f32 spawnRadius{20.0f}; /**< How far from the focus a body may appear. */
};

/**
 * @class LivingLayer
 * @brief Population, bodies and the substrate they read.
 */
class LivingLayer {
public:
    /**
     * @brief Opens the scent window and seeds the food web from a recipe.
     *
     * It configures the one HerdParams this layer runs on: the systems hold a
     * reference to it, so it is configured here rather than rebuilt each step.
     *
     * @param params The budgets the host can afford.
     * @param recipe The ecology to seed.
     * @param seed The random seed for reproducible results.
     */
    void configure(const LivingLayerParams &params, const ecology::LivingRecipe &recipe, core::u32 seed);

    /**
     * @brief Rebuilds the food web, with the producer capacity taken from the world.
     *
     * The producer's capacity is the one value a document cannot state: it is how
     * much vegetation this seed actually grew, counted. Everything else is the
     * cartridge's.
     *
     * Zero means "the world cannot count its plants yet" — a streamed world has no
     * total, only what is resident — and then the recipe's own capacity stands.
     * Overriding it with zero starves the producers on the first tick, and a food web
     * whose bottom is empty inverts: three grazers, forty-four hunters, which is what
     * the screen showed.
     */
    void seedWeb(core::u32 standingPlants);

    void openScent(core::u32 span, core::u32 layers) { _scent.open(span, layers); }

    /**
     * @brief Binds the herd's bodies to the registry they live in.
     *
     * Must happen before the first @ref spawn: a body is an entity, and a herd
     * with no registry has nowhere to put one.
     */
    void bind(ecs::Registry &registry) noexcept { _herd.bind(registry); }

    /**
     * @brief Where the scent window currently sits, for the creature systems.
     */
    [[nodiscard]] systems::CreatureFieldView fieldView() const noexcept;

    [[nodiscard]] ai::ScentWindow &scent() noexcept { return _scent; }
    [[nodiscard]] const ai::ScentWindow &scent() const noexcept { return _scent; }
    [[nodiscard]] ecology::Herd &herd() noexcept { return _herd; }
    [[nodiscard]] const ecology::Herd &herd() const noexcept { return _herd; }
    [[nodiscard]] ecology::TrophicWeb &web() noexcept { return _web; }
    [[nodiscard]] const ecology::TrophicWeb &web() const noexcept { return _web; }
    [[nodiscard]] const ecology::LivingRecipe &recipe() const noexcept { return _recipe; }
    [[nodiscard]] core::u32 grazedCount() const noexcept { return _grazed; }
    [[nodiscard]] const LivingLayerParams &params() const noexcept { return _params; }
    void countGrazed() noexcept { ++_grazed; }

    /** No such species. */
    static constexpr core::u32 kNoSpecies = 0xFFFFFFFFu;

    /**
     * @brief Which web species the nth BODIED species is.
     *
     * Producers are skipped, and that is a correction: bodiesFor used to index the
     * web directly, so with the canonical four-level recipe — whose species 0 is
     * grass — the count of grazers on screen was derived from the GRASS population
     * and the count of hunters from the deer. It survived because the result still
     * looks like a food web ratio (many grazers, few hunters), so the screen was
     * right for the wrong reason, and no gate goes through this class.
     *
     * A producer has no bodies because it already has a representation: vegetation
     * is drawn as plants.
     *
     * @param bodied Index among the species that have bodies.
     * @return The web species index, or @ref kNoSpecies.
     */
    [[nodiscard]] core::u32 webIndexOfBodied(core::u32 bodied) const noexcept;

    /**
     * @brief How many bodies a species should have on screen.
     *
     * The DOCUMENT decides the ratio and the HOST decides the ceiling.
     * @c LivingRecipe::headPerBody is what a cartridge declares one drawn body to
     * stand for; it travelled through the codec, the baker and two tests and was
     * read by nothing, while this function used a budget share instead and the map
     * viewer hard-coded a division by two. Three answers to one question.
     *
     * The budget is applied as a proportional CAP over the whole herd, not as the
     * ratio: clamping each species to the same ceiling independently gave
     * twenty-four grazers and twenty-four hunters out of a web that says five to
     * one, which reads as a plague of predators.
     *
     * @param species Index among the species that have bodies.
     * @return The number of bodies this species should have.
     */
    [[nodiscard]] core::u32 bodiesFor(core::u32 species) const noexcept;

private:
    /**
     * @brief Bodies a head count is worth before the budget caps it.
     *
     * At or below its refuge the species is gone as far as the model is concerned,
     * and drawing a body for it would contradict the census. Above its refuge it
     * exists — and a world where existing is invisible is not showing the model, so
     * it gets at least one body. This is the floor that was learned the hard way: at
     * one body per ten head, a population of seven floored to zero and the map stayed
     * empty while the HUD cheerfully reported a working ecosystem. The spawning, the
     * flocking and the drawing were all correct and all ran on an empty list.
     */
    [[nodiscard]] core::u32 rawBodiesFor(core::u32 webIndex) const noexcept;

public:
    /**
     * @brief Creates one body of a species somewhere it can stand.
     *
     * The archetype is the species; the genome is this individual. Mutation is what
     * makes a herd look like animals rather than like copies.
     *
     * The body is an ENTITY and every part of it is a component: Position, Velocity
     * and AABB are what the physics solver already understands, so a creature stops
     * being a special case the moment it carries them; Genome and Creature are what
     * the reflection registry knows how to describe, so a document — and an
     * intelligence — can name one; and Heading is what a walker advances along, which
     * is why it cannot stay outside the registry the walker lives in.
     *
     * @param placeAt (attempt, outX, outZ) -> bool: proposes a position and says
     *                whether an animal may stand there. Twenty-four attempts, then
     *                give up — a world can legitimately have nowhere to put one.
     */
    template <typename PlaceAt> bool spawn(math::Random &random, core::u32 species, PlaceAt &&placeAt);

    /**
     * @brief Brings the bodies in line with the census, keeping the ratio.
     * @param tick The current tick.
     * @param placeAt The function to place an animal at a position.
     */
    template <typename PlaceAt> void reconcile(core::u32 tick, PlaceAt &&placeAt);

    /**
     * @brief How this layer's animals move and what they smell.
     *
     * ONE instance, handed to the systems by reference. The moment there were two
     * — a local one built inside the step and another beside it for the systems —
     * a palate changed in one place would have stopped matching what actually ran.
     */
    [[nodiscard]] const ecology::HerdParams &herdParams() const noexcept { return _herdParams; }

    /**
     * @brief Fills a freshly created creature entity's components.
     *
     * The AABB is sized by the genome, so a big animal is a big obstacle. Mass is left
     * off deliberately: a creature with no Mass is static to the solver, and giving one
     * to a body the locomotion system already walks would have two things moving it.
     * The facing is seeded from the initial velocity, not from a fixed direction: a
     * herd that all faces north walks north together on its first tick.
     *
     * @return false when the entity could not be reached, so a caller can undo the
     *         creation instead of leaving a body with no position in the world.
     */
    [[nodiscard]] static bool writeBody(ecs::Registry &registry, ecs::EntityId entity, core::u32 species,
                                        core::u32 identity, const ecology::Genome &genome, math::Fixed32 x,
                                        math::Fixed32 z, math::Fixed32 vx, math::Fixed32 vz);

    /**
     * @brief Sets the producer population from what is actually standing.
     *
     * Index zero is the producer by convention of the recipe's ordering, and the
     * count is a FACT the world measures rather than a number the model integrates:
     * grazing a valley bare has to move it, and it only does if it is a census.
     */
    void setProducerPopulation(core::u32 standing) noexcept;

    /**
     * @brief One tick of the substrate: the trails diffuse and evaporate.
     */
    void stepScent() { _scent.field().step(_recipe.stigmergy); }

    /**
     * @brief Integrates the food web. Called every @c webPeriod ticks.
     * @param steps The number of steps to integrate.
     */
    void stepWeb(core::u32 steps = 1u) { _web.step(steps); }

private:
    LivingLayerParams _params{};
    ecology::LivingRecipe _recipe{ecology::parityLivingRecipe()};
    ecology::Herd _herd;
    ecology::TrophicWeb _web{};
    ai::ScentWindow _scent;
    ecology::HerdParams _herdParams{};
    core::u32 _seed{1337u};
    core::u32 _heredity{1u};
    core::u32 _nextId{1u};
    core::u32 _grazed{0u};
};

} // namespace lpl::engine

#    include "LivingLayer.inl"

#endif // LPL_ENGINE_LIVING_LAYER_HPP
