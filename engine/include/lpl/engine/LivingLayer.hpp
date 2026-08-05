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
#    include <lpl/engine/systems/CreatureSystems.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/math/Random.hpp>

namespace lpl::engine {

/**
 * @struct LivingLayerParams
 * @brief The budgets a host can afford for the living layer.
 */
struct LivingLayerParams {
    core::u32 maxBodies{48u};     ///< Bodies on screen, all species together.
    core::u32 speciesCount{2u};   ///< Species that get bodies (grazer, hunter).
    core::u32 scentSpan{64u};     ///< Cells across the pheromone window.
    /// Stigmergy channels. Six, because that is how many @c ai::ScentChannel names
    /// and the herd now reads them by meaning: plant, herbivore, carnivore, terror,
    /// kin, pheromone. Two was enough while every animal climbed one hard-coded
    /// channel — which is exactly the bug that made a grazer follow its own scent.
    core::u32 scentLayers{6u};
    core::u32 webPeriod{60u};     ///< Ticks between two food-web integrations.
    core::f32 spawnRadius{20.0f}; ///< How far from the focus a body may appear.
};

/**
 * @class LivingLayer
 * @brief Population, bodies and the substrate they read.
 */
class LivingLayer {
public:
    /**
     * @brief Opens the scent window and seeds the food web from a recipe.
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

    /// Where the scent window currently sits, for the creature systems.
    [[nodiscard]] systems::CreatureFieldView fieldView() const noexcept
    {
        return systems::CreatureFieldView{_scent.originX(), _scent.originZ(), _scent.field().width(),
                                          _scent.field().depth()};
    }

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

    /// No such species.
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
    /// Bodies a head count is worth before the budget caps it.
    [[nodiscard]] core::u32 rawBodiesFor(core::u32 webIndex) const noexcept;

public:

    /**
     * @brief Creates one body of a species somewhere it can stand.
     *
     * @param placeAt (attempt, outX, outZ) -> bool: proposes a position and says
     *                whether an animal may stand there. Twenty-four attempts, then
     *                give up — a world can legitimately have nowhere to put one.
     */
    template <typename PlaceAt> bool spawn(math::Random &random, core::u32 species, PlaceAt &&placeAt)
    {
        if (_herd.size() >= _params.maxBodies || _herd.registry() == nullptr)
            return false;

        const core::u32 identity = _nextId++;

        // The archetype is the species; the genome is this individual. Mutation is
        // what makes a herd look like animals rather than like copies.
        ecology::Genome stock{};
        if (species == 1u)
        {
            stock.size = math::Fixed32::fromFloat(1.4f);
            stock.maxSpeed = math::Fixed32::fromFloat(5.0f);
        }
        else
        {
            stock.size = math::Fixed32::fromFloat(0.9f);
            stock.maxSpeed = math::Fixed32::fromFloat(3.5f);
        }
        const ecology::Genome genome = ecology::mutate(stock, 8u, 0.18f, _heredity);

        for (core::u32 attempt = 0u; attempt < 24u; ++attempt)
        {
            math::Fixed32 x{};
            math::Fixed32 z{};
            if (!placeAt(random, attempt, x, z))
                continue;
            // The body is an ENTITY and every part of it is a component: Position,
            // Velocity and AABB are what the physics solver already understands, so
            // a creature stops being a special case the moment it carries them;
            // Genome and Creature are what the reflection registry knows how to
            // describe, so a document — and an intelligence — can name one; and
            // Heading is what a walker advances along, which is why it cannot stay
            // outside the registry the walker lives in.
            ecs::Archetype archetype;
            archetype.add(ecs::ComponentId::Position);
            archetype.add(ecs::ComponentId::Velocity);
            archetype.add(ecs::ComponentId::AABB);
            archetype.add(ecs::ComponentId::Genome);
            archetype.add(ecs::ComponentId::Creature);
            archetype.add(ecs::ComponentId::Heading);

            ecs::Registry &registry = *_herd.registry();
            auto created = registry.createEntity(archetype);
            if (!created.has_value())
                return false;
            const ecs::EntityId entity = created.value();

            const math::Fixed32 vx = random.unit() - math::Fixed32::half();
            const math::Fixed32 vz = random.unit() - math::Fixed32::half();

            if (!writeBody(registry, entity, species, identity, genome, x, z, vx, vz))
            {
                (void) registry.destroyEntity(entity);
                return false;
            }
            _herd.add(entity);
            return true;
        }
        return false;
    }

    /**
     * @brief Brings the bodies in line with the census, keeping the ratio.
     * @param tick The current tick.
     * @param placeAt The function to place an animal at a position.
     */
    template <typename PlaceAt> void reconcile(core::u32 tick, PlaceAt &&placeAt)
    {
        math::Random stock{_seed ^ (0xB0D533u + tick)};
        _herd.reconcile(
            _params.speciesCount, [this](core::u32 species) { return bodiesFor(species); },
            [this, &stock, &placeAt](core::u32 species) { return spawn(stock, species, placeAt); });
    }

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
     * @return false when the entity could not be reached, so a caller can undo the
     *         creation instead of leaving a body with no position in the world.
     */
    [[nodiscard]] static bool writeBody(ecs::Registry &registry, ecs::EntityId entity, core::u32 species,
                                        core::u32 identity, const ecology::Genome &genome, math::Fixed32 x,
                                        math::Fixed32 z, math::Fixed32 vx, math::Fixed32 vz)
    {
        auto ref = registry.resolve(entity);
        if (!ref.has_value())
            return false;
        for (const auto &part : registry.partitions())
        {
            if (!part || !part->archetype().has(ecs::ComponentId::Creature))
                continue;
            const auto &chunks = part->chunks();
            if (ref.value().chunkIndex >= static_cast<core::u32>(chunks.size()) || !chunks[ref.value().chunkIndex])
                continue;
            auto &chunk = *chunks[ref.value().chunkIndex];
            const auto entities = chunk.entities();
            if (ref.value().localIndex >= entities.size() || entities[ref.value().localIndex] != entity)
                continue;
            const core::u32 slot = ref.value().localIndex;

            if (auto *p = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position)))
                p[slot] = {x, math::Fixed32{}, z};
            if (auto *v = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Velocity)))
                v[slot] = {vx, math::Fixed32{}, vz};
            // Sized by the genome, so a big animal is a big obstacle. Mass is left
            // off deliberately: a creature with no Mass is static to the solver,
            // and giving one to a body the locomotion system already walks would
            // have two things moving it.
            if (auto *b = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
                b[slot] = {genome.size * math::Fixed32::half(), genome.size, genome.size * math::Fixed32::half()};
            if (auto *g = static_cast<ecology::Genome *>(chunk.writeComponent(ecs::ComponentId::Genome)))
                g[slot] = genome;
            if (auto *c = static_cast<core::u32 *>(chunk.writeComponent(ecs::ComponentId::Creature)))
            {
                c[slot * 2u] = species;
                c[slot * 2u + 1u] = identity;
            }
            // Facing seeded from the initial velocity, not from a fixed direction:
            // a herd that all faces north walks north together on its first tick.
            if (auto *h = static_cast<math::Fixed32 *>(chunk.writeComponent(ecs::ComponentId::Heading)))
            {
                h[slot * 2u] = vx;
                h[slot * 2u + 1u] = vz;
            }
            return true;
        }
        return false;
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

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/engine/LivingLayer.inl>

#endif // LPL_ENGINE_LIVING_LAYER_HPP
