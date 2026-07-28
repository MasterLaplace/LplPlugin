/**
 * @file Social.hpp
 * @brief Who a creature thinks you are, and why it stopped trusting you.
 *
 * The step beyond friend-or-foe is that the opinion is **asymmetric and
 * remembered**. A creature builds a mental model of another when it sees it, and
 * that model keeps evolving after line of sight is lost — extrapolated from the
 * last known velocity, decaying in confidence. What a creature reacts to is its
 * model, not the truth, which is where hiding, stalking and being wrong all come
 * from.
 *
 * On top of that sits reputation, and it is what makes a player's actions
 * accumulate. Attack one member of a faction unprovoked and the faction's opinion
 * of you shifts, slowly, from `Ignores` toward `Attacks` or `Afraid`. Nothing
 * scripted the grudge.
 *
 * The relationships are then filtered by personality (@ref Personality.hpp): the
 * same remembered slight makes an aggressive creature charge and a nervous one
 * flee. One memory, six modulations.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_SOCIAL_HPP
#    define LPL_AI_SOCIAL_HPP

#    include <lpl/ai/Personality.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ai {

/**
 * @enum Attitude
 * @brief The archetype of one creature's opinion of another.
 */
enum class Attitude : core::u8 {
    Ignores = 0, ///< No opinion.
    Eats,        ///< Prey.
    Afraid,      ///< Predator, or something that hurt it.
    Antagonises, ///< Rival, worth harassing but not killing.
    Pack,        ///< Kin or ally.
    Attacks      ///< Actively hostile.
};

/**
 * @struct Opinion
 * @brief What A thinks of B, and how sure it is.
 */
struct Opinion {
    core::u32 subject{0u}; ///< Who this is about.
    Attitude attitude{Attitude::Ignores};
    math::Fixed32 intensity{}; ///< Strength of the attitude, in [0, 1].

    /**
     * @brief How much the model is still trusted, in [0, 1].
     *
     * Decays while the subject is out of sight. Confidence is what separates
     * "I know where it is" from "I think it went that way", and it is why a
     * creature can be made to search the wrong place — the interesting failure.
     */
    math::Fixed32 confidence{};

    core::u32 lastSeenCell{0u}; ///< Where it was last observed.
    math::Fixed32 lastVx{};     ///< Its velocity then; the extrapolation's basis.
    math::Fixed32 lastVz{};
};

/**
 * @class RelationshipTracker
 * @brief Per-creature memory of other creatures, and per-faction reputation.
 */
class RelationshipTracker {
public:
    /**
     * @brief Records a sighting, creating or refreshing the model.
     *
     * @param observer Who is looking.
     * @param subject  What was seen.
     * @param cell     Where.
     * @param vx       Its velocity along X.
     * @param vz       Its velocity along Z.
     * @param attitude What the observer makes of it.
     */
    void observe(core::u32 observer, core::u32 subject, core::u32 cell, math::Fixed32 vx, math::Fixed32 vz,
                 Attitude attitude);

    /**
     * @brief Ages every model: confidence falls, position is extrapolated.
     *
     * The extrapolation is what makes a lost target searchable rather than simply
     * forgotten. It is deliberately naive — last velocity, straight line — because
     * the interesting behaviour is the creature being WRONG about where the target
     * went, and a good predictor would remove it.
     *
     * @param width  Grid width, to move a cell index.
     * @param depth  Grid depth.
     * @param decay  Confidence retained per tick, in (0, 1].
     * @return Number of models that dropped below usefulness and were forgotten.
     */
    core::u32 tick(core::u32 width, core::u32 depth, math::Fixed32 decay);

    /**
     * @brief What @p observer currently believes about @p subject.
     * @param observer Who is remembering.
     * @param subject  Who is remembered.
     * @param out      Receives the opinion.
     * @return true when a model exists.
     */
    [[nodiscard]] bool opinion(core::u32 observer, core::u32 subject, Opinion &out) const;

    /**
     * @brief Records an unprovoked attack, shifting a whole faction's view.
     *
     * The reputation mechanism: the victim's faction, not just the victim,
     * remembers. This is what makes a player's history follow them, and it costs
     * one number per faction rather than a model per member.
     *
     * @param faction  The victim's faction.
     * @param attacker Who did it.
     * @param severity How badly, in [0, 1].
     */
    void recordAggression(core::u32 faction, core::u32 attacker, math::Fixed32 severity);

    /**
     * @brief How a faction feels about someone.
     * @param faction The faction.
     * @param subject Who they are judging.
     * @return Reputation in [-1, 1]; negative is hostile.
     */
    [[nodiscard]] math::Fixed32 reputation(core::u32 faction, core::u32 subject) const;

    /**
     * @brief The attitude a creature acts on, personality included.
     *
     * The remembered attitude is only half the answer. An aggressive creature
     * escalates `Antagonises` into `Attacks`; a sympathetic one de-escalates it.
     * Same memory, different behaviour — which is the point of having both
     * systems rather than one.
     *
     * @param remembered What the model says.
     * @param traits     The creature's temperament.
     * @param intensity  How strongly it is felt.
     * @return The attitude to act on.
     */
    [[nodiscard]] static Attitude effectiveAttitude(Attitude remembered, const PersonalityTraits &traits,
                                                    math::Fixed32 intensity);

    [[nodiscard]] core::u32 modelCount() const noexcept { return static_cast<core::u32>(_models.size()); }

    /// @brief FNV-1a fold of every model and reputation, for determinism checks.
    [[nodiscard]] core::u32 fold() const;

private:
    struct Model {
        core::u32 observer;
        Opinion opinion;
    };
    struct Reputation {
        core::u32 faction;
        core::u32 subject;
        math::Fixed32 value;
    };

    lpl::pmr::vector<Model> _models;
    lpl::pmr::vector<Reputation> _reputations;
};

} // namespace lpl::ai

#endif // LPL_AI_SOCIAL_HPP
