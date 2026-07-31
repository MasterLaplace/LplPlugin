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
 * terrain: where an animal may stand and what it eats arrive as callbacks.
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
#    include <lpl/ecology/Herd.hpp>
#    include <lpl/ecology/LivingRecipe.hpp>
#    include <lpl/ecology/Populations.hpp>
#    include <lpl/ecology/Vegetation.hpp>
#    include <lpl/procgen/Random.hpp>

namespace lpl::engine {

/**
 * @struct LivingLayerParams
 * @brief The budgets a host can afford for the living layer.
 */
struct LivingLayerParams {
    core::u32 maxBodies{48u};     ///< Bodies on screen, all species together.
    core::u32 speciesCount{2u};   ///< Species that get bodies (grazer, hunter).
    core::u32 scentSpan{64u};     ///< Cells across the pheromone window.
    core::u32 scentLayers{2u};    ///< Pheromone channels.
    core::u32 webPeriod{60u};     ///< Ticks between two food-web integrations.
    core::f32 spawnRadius{20.0f}; ///< How far from the focus a body may appear.
};

/**
 * @class LivingLayer
 * @brief Population, bodies and the substrate they read.
 */
class LivingLayer {
public:
    /** @brief Opens the scent window and seeds the food web from a recipe. */
    void configure(const LivingLayerParams &params, const ecology::LivingRecipe &recipe, core::u32 seed);

    /**
     * @brief Rebuilds the food web, with the producer capacity taken from the world.
     *
     * The producer's capacity is the one value a document cannot state: it is how
     * much vegetation this seed actually grew, counted. Everything else is the
     * cartridge's.
     */
    void seedWeb(core::u32 standingPlants);

    void openScent(core::u32 span, core::u32 layers) { _scent.open(span, layers); }

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

    /**
     * @brief How many bodies a species should have on screen.
     *
     * The whole herd is scaled to the budget, not each species to a cap: see the file
     * comment for what the other version looked like.
     */
    [[nodiscard]] core::u32 bodiesFor(core::u32 species) const noexcept;

    /**
     * @brief Creates one body of a species somewhere it can stand.
     *
     * @param placeAt (attempt, outX, outZ) -> bool: proposes a position and says
     *                whether an animal may stand there. Twenty-four attempts, then
     *                give up — a world can legitimately have nowhere to put one.
     */
    template <typename PlaceAt> bool spawn(procgen::Random &random, core::u32 species, PlaceAt &&placeAt)
    {
        if (_herd.size() >= _params.maxBodies)
            return false;

        ecology::HerdMember member;
        member.id = _nextId++;
        member.species = species;

        // The archetype is the species; the genome is this individual. Mutation is
        // what makes a herd look like animals rather than like copies.
        ecology::Genome archetype{};
        if (species == 1u)
        {
            archetype.size = math::Fixed32::fromFloat(1.4f);
            archetype.maxSpeed = math::Fixed32::fromFloat(5.0f);
        }
        else
        {
            archetype.size = math::Fixed32::fromFloat(0.9f);
            archetype.maxSpeed = math::Fixed32::fromFloat(3.5f);
        }
        member.genome = ecology::mutate(archetype, 8u, 0.18f, _heredity);

        for (core::u32 attempt = 0u; attempt < 24u; ++attempt)
        {
            math::Fixed32 x{};
            math::Fixed32 z{};
            if (!placeAt(random, attempt, x, z))
                continue;
            member.body.x = x;
            member.body.z = z;
            member.body.vx = random.unit() - math::Fixed32::half();
            member.body.vz = random.unit() - math::Fixed32::half();
            member.heading = member.body.vx;
            member.headingZ = member.body.vz;
            _herd.add(member);
            return true;
        }
        return false;
    }

    /** @brief Brings the bodies in line with the census, keeping the ratio. */
    template <typename PlaceAt> void reconcile(core::u32 tick, PlaceAt &&placeAt)
    {
        procgen::Random stock{_seed ^ (0xB0D533u + tick)};
        _herd.reconcile(
            _params.speciesCount, [this](core::u32 species) { return bodiesFor(species); },
            [this, &stock, &placeAt](core::u32 species) { return spawn(stock, species, placeAt); });
    }

    /**
     * @brief One tick of the herd: flocking, scent, food, movement.
     *
     * The herd's parameters are built from this layer's own budgets rather than by
     * the caller: how a grazer flocks against how a hunter does is a property of the
     * ECOLOGY, not of the game that hosts it. What the caller supplies is where an
     * animal may stand and what happens where it eats.
     */
    template <typename ToFieldCell, typename Walkable, typename Graze>
    void stepHerd(math::Fixed32 step, ToFieldCell &&toFieldCell, Walkable &&walkable, Graze &&graze)
    {
        ecology::HerdParams params;
        params.speciesCount = _params.speciesCount;
        params.step = step;
        _herd.step(params, _scent.field(), toFieldCell, walkable, graze);
    }

    /**
     * @brief Sets the producer population from what is actually standing.
     *
     * Index zero is the producer by convention of the recipe's ordering, and the
     * count is a FACT the world measures rather than a number the model integrates:
     * grazing a valley bare has to move it, and it only does if it is a census.
     */
    void setProducerPopulation(core::u32 standing) noexcept
    {
        if (!_web.species.empty() && standing != 0u)
            _web.species[0].population = math::Fixed32::fromInt(static_cast<core::i32>(standing));
    }

    /** @brief One tick of the substrate: the trails diffuse and evaporate. */
    void stepScent() { _scent.field().step(_recipe.stigmergy); }

    /** @brief Integrates the food web. Called every @c webPeriod ticks. */
    void stepWeb(core::u32 steps = 1u) { _web.step(steps); }

private:
    LivingLayerParams _params{};
    ecology::LivingRecipe _recipe{ecology::parityLivingRecipe()};
    ecology::Herd _herd;
    ecology::TrophicWeb _web{};
    ai::ScentWindow _scent;
    core::u32 _seed{1337u};
    core::u32 _heredity{1u};
    core::u32 _nextId{1u};
    core::u32 _grazed{0u};
};

} // namespace lpl::engine

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/engine/LivingLayer.inl>

#endif // LPL_ENGINE_LIVING_LAYER_HPP
