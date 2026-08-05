/**
 * @file Herd.hpp
 * @brief Which entities are a world's animals, and how many of each there should be.
 *
 * This used to be where an animal LIVED: a struct holding a flocked body, a genome
 * and an identity, plus a step() that flocked it, steered it by scent, fed it and
 * walked it. Every one of those is now either a component in the registry or a
 * system over it (engine::systems::Flocking / ScentSteering / Grazing / Locomotion),
 * which is what makes an animal something the physics, a document and an
 * intelligence can all see.
 *
 * What is left is the one thing a component cannot say: WHICH entities are this
 * world's herd. That is a set, not per-body state, so holding it here duplicates
 * nothing — and it is what the census reconciles against, because the population
 * model is the authority and the bodies are its sample.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-08-04
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECOLOGY_HERD_HPP
#    define LPL_ECOLOGY_HERD_HPP

#    include <lpl/ai/StigmergyField.hpp>
#    include <lpl/ecs/Component.hpp>
#    include <lpl/ecs/Entity.hpp>
#    include <lpl/ecs/Partition.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ecology {

/// Trophic roles the herd distinguishes. Two is what the sim uses today.
inline constexpr core::u32 kMaxHerdSpecies = 4u;

/**
 * @struct SpeciesScent
 * @brief What one species leaves behind, and what it steers by.
 *
 * Before this existed, the herd read channel 1 uphill for EVERY animal. Nothing
 * deposited, four of the six named channels were never read, and a herbivore
 * climbed the herbivore scent — it was attracted to itself instead of fleeing the
 * predator hunting it. The bug survived because the herd was exercised by no
 * gate: it drives the playable world, and the living-parity gate steps boids
 * directly.
 */
struct SpeciesScent {
    static constexpr core::u32 kNoDeposit = 0xFFFFFFFFu;

    /// Where this species leaves its mark; @ref kNoDeposit to leave none.
    core::u32 depositChannel{kNoDeposit};
    math::Fixed32 depositAmount{math::Fixed32::one()};
    ai::ScentPalate palate{}; ///< What it is drawn to and what it flees.
};

/**
 * @brief Whether a species eats plants where it stands.
 *
 * Read off the scent declaration rather than off a species INDEX. "Species 0 is
 * the grazer" is a convention several files repeat and none enforces; "it smells
 * of herbivore, so it eats plants" is a fact stated once, in data, by whoever
 * declared the ecology.
 *
 * @param scent The species' scent declaration.
 * @return True when this species forages.
 */
[[nodiscard]] constexpr bool isHerbivore(const SpeciesScent &scent) noexcept
{
    return scent.depositChannel == static_cast<core::u32>(ai::ScentChannel::Herbivore);
}

/**
 * @brief The default ecology: a grazer, a hunter, and the flanking that follows.
 *
 * Encirclement is not scripted here, and that is the claim worth testing: hunters
 * are pulled toward the herbivore scent and pushed off each other's, so a pack
 * that all wanted the same cell spreads around the prey instead of stacking
 * behind it. Nothing in the code says "flank".
 */
[[nodiscard]] inline SpeciesScent defaultGrazerScent() noexcept;
[[nodiscard]] inline SpeciesScent defaultHunterScent() noexcept;

/**
 * @struct HerdParams
 * @brief How a species moves, per trophic role.
 */
struct HerdParams {
    core::u32 speciesCount{2u};
    core::f32 separationHunter{1.1f};
    core::f32 separationGrazer{0.9f};
    core::f32 alignmentHunter{0.5f};
    core::f32 alignmentGrazer{0.8f};
    core::f32 cohesionHunter{0.25f};
    core::f32 cohesionGrazer{0.5f};
    core::i32 neighbourHunter{10};
    core::i32 neighbourGrazer{6};
    /// Below this, push apart. A pack holds a looser formation than a herd, and one
    /// radius for both made the wolves move like a shoal — which is exactly what a
    /// pack must not look like.
    math::Fixed32 separationHunterRadius{math::Fixed32::fromFloat(2.5f)};
    math::Fixed32 separationGrazerRadius{math::Fixed32::fromFloat(1.6f)};
    /// How hard a scent gradient pulls, before personality scales it.
    math::Fixed32 scentPull{math::Fixed32::fromFloat(0.06f)};

    /// Per-species olfaction, indexed by the Creature component's species.
    SpeciesScent scent[kMaxHerdSpecies]{};
    math::Fixed32 step{math::Fixed32::fromRaw(1092)}; ///< One tick, in seconds (60 Hz).
};

/**
 * @brief Applies @ref defaultGrazerScent / @ref defaultHunterScent to a params set.
 * @param params The parameters to fill.
 */
inline void applyDefaultScents(HerdParams &params) noexcept;

/**
 * @class Herd
 * @brief The roster of a world's animal bodies, and the census over it.
 */
class Herd {
public:
    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_bodies.size()); }
    [[nodiscard]] bool empty() const noexcept { return _bodies.empty(); }
    [[nodiscard]] ecs::EntityId at(core::u32 index) const noexcept { return _bodies[index]; }
    void add(ecs::EntityId entity) { _bodies.push_back(entity); }

    /**
     * @brief Binds the herd to the registry its bodies live in.
     *
     * Must be called before the first body is added. A herd with no registry has
     * no bodies to own, and says so by doing nothing rather than by reading a null.
     */
    void bind(ecs::Registry &registry) noexcept { _registry = &registry; }
    [[nodiscard]] ecs::Registry *registry() const noexcept { return _registry; }

    /**
     * @brief Destroys every body and empties the roster.
     *
     * Destroying is the point. Dropping the roster alone left the entities alive
     * in the registry, and since the renderer draws what is IN THE WORLD rather
     * than what a container remembers, regenerating a world left its previous
     * animals standing in the new one for good.
     */
    void clear();

    /**
     * @brief Animals of one species, for a census or a HUD.
     *
     * Counted from the Creature component, not from a cached role: the component
     * is the single place that says what an animal is.
     *
     * @param species The species to count.
     * @return The number of animals of the specified species.
     */
    [[nodiscard]] core::u32 countSpecies(core::u32 species) const noexcept;

    /**
     * @brief Destroys one animal of a species; used to reconcile with the census.
     * @param species The species of the animal to remove.
     * @return True when one was found and removed.
     */
    bool removeOne(core::u32 species) noexcept;

    /**
     * @brief Brings the number of bodies in line with what a census says exists.
     *
     * The population model is the authority — it integrates births and deaths — and
     * the bodies are its view. Letting the bodies drift from it is how a species goes
     * extinct in the numbers while still walking around on screen.
     *
     * @param spawn (species) -> bool, creating one animal; false when it could not.
     */
    template <typename CountFor, typename Spawn>
    void reconcile(core::u32 speciesCount, CountFor &&countFor, Spawn &&spawn)
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

    /**
     * @brief An entity's Creature component: {species, id}, or nullptr.
     *
     * Public because the systems that walk the registry by chunk do not need it,
     * but a HUD or a test that holds an entity id does — and a second copy of this
     * lookup is exactly what this file exists to prevent.
     *
     * Reads the WRITE side, like every creature system: see CreatureSystems.cpp
     * for why a world whose systems are all PrePhysics never publishes a front
     * buffer at all.
     *
     * @param registry The registry to look in.
     * @param entity   The entity to resolve.
     * @return Pointer to two u32 (species, id), or nullptr when unreachable.
     */
    [[nodiscard]] static const core::u32 *creatureOf(ecs::Registry &registry, ecs::EntityId entity) noexcept;

private:
    lpl::pmr::vector<ecs::EntityId> _bodies; ///< Which entities this herd owns.
    ecs::Registry *_registry{nullptr};
};

} // namespace lpl::ecology

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/ecology/Herd.inl>

#endif // LPL_ECOLOGY_HERD_HPP
