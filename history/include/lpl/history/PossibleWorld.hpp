/**
 * @file PossibleWorld.hpp
 * @brief One hypothesis set, one World instance.
 *
 * Contextual graph layering, realised with machinery that already exists: a
 * Server hosts N Worlds with isolated registries, so the world-according-to-source-A
 * and the world-according-to-archaeology are two instances of the same engine fed
 * different constraints. Comparing them is comparing two folds.
 *
 * The class here is the half that turns a corpus into a timeline: which sources this
 * world listens to, how it fuses agreement, and what it does when two of them
 * contradict each other. The other half — hosting N of them — is
 * `lpl::engine::Server` and is not re-implemented here.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_POSSIBLEWORLD_HPP
#    define LPL_LPL_HISTORY_POSSIBLEWORLD_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/history/Fact.hpp>
#    include <lpl/history/Timeline.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::history {

/**
 * @struct Corpus
 * @brief Facts, and what is known about the sources that assert them.
 */
struct Corpus {
    lpl::pmr::vector<Fact> facts{};
    lpl::pmr::vector<SourceProfile> sources{};

    /**
     * @brief Looks a source up.
     * @param id      Source identifier.
     * @param outProfile Receives it.
     * @return false when the corpus does not describe that source.
     */
    [[nodiscard]] bool sourceProfile(core::u32 id, SourceProfile &outProfile) const noexcept;
};

/**
 * @struct WorldView
 * @brief Which sources a possible world listens to, and how hard it holds them.
 */
struct WorldView {
    /**
     * @brief Sources this world admits; empty means all of them.
     *
     * This is what makes a possible world possible: the world-according-to-Herodotus
     * is the same engine, the same recipe and the same systems, listening to one
     * source. Nothing forks.
     */
    lpl::pmr::vector<core::u32> admittedSources{};

    /**
     * @brief Below this fused confidence a fact is admitted but only SCORED.
     *
     * The line between "the run must honour this" and "the run is measured against
     * it". Putting it on confidence rather than on source is what stops a strong
     * claim from a weak source being ignored, and a weak claim from a strong source
     * being obeyed.
     *
     * 0.60 by default.
     */
    math::Fixed32 seedThreshold{math::Fixed32::fromRaw(39322)};

    /**
     * @brief At or above this, a fact is FORCED for its whole window.
     *
     * High on purpose. Every forced fact is one the run did not have to explain.
     *
     * 0.92 by default.
     */
    math::Fixed32 forceThreshold{math::Fixed32::fromRaw(60293)};

    TrustWeights weights{}; /**< How this world scores its sources. */
};

/**
 * @struct FusionReport
 * @brief What building a timeline out of a corpus had to decide.
 */
struct FusionReport {
    core::u32 admitted{0u};       /**< Facts this world listened to. */
    core::u32 rejectedSource{0u}; /**< Facts dropped because their source is not admitted. */
    core::u32 fused{0u};          /**< Facts whose confidence rose because others agreed. */
    core::u32 contradictions{0u}; /**< Pairs that cannot both hold. */
    core::u32 demoted{0u};        /**< Claims a contradiction pushed down — never erased. */
    core::u32 seeded{0u};         /**< Constraints of kind Seed. */
    core::u32 forced{0u};         /**< Constraints of kind Force. */
    core::u32 scored{0u};         /**< Constraints of kind Score. */
};

/**
 * @brief Turns a corpus into the timeline THIS world believes.
 *
 * Three steps, in order, and the order matters:
 *
 * 1. **Admission.** Facts from sources this world does not listen to are dropped.
 *    A claim's weight is its own sigma tempered by how much its source is worth
 *    believing: a confident assertion in a panegyric is a confident assertion in a
 *    panegyric. The source is scored against THIS claim's date, not as a whole: a
 *    source that states a composition window is ten years from one of its claims and
 *    eight hundred from another, and one number for both is right about at most one.
 *    Where no window is stated the per-claim score falls back on whatever distance the
 *    profile declares, so a corpus that knows nothing new folds exactly as before.
 * 2. **Fusion.** Independent sources asserting the same triple raise each other's
 *    confidence — SIM-020's `1 - (1-p)(1-q)`, weighted by how much each source is
 *    worth believing. The same source agreeing with itself is not evidence, and the
 *    windows have to OVERLAP: two sources saying a settlement existed — one in 1200
 *    and one in 1900 — are not agreeing with each other, they are describing two
 *    different states of the world, and fusing them would manufacture confidence out
 *    of the mere reuse of a name.
 * 3. **Contradiction.** Where two claims cannot both hold, the LESS supported one is
 *    demoted, never deleted. That is the whole point of SIM-022: the consensus view
 *    is the default, and the minority version stays reachable, so how a myth was
 *    built can still be traced. A pipeline that erased the loser would destroy the
 *    evidence for its own decision. The loser loses ground in proportion to how far
 *    ahead the other is, and keeps a floor of 0.05: a demoted claim that reached zero
 *    would be indistinguishable from one nobody ever made.
 *
 * The verdict then says what the run must do about each claim: Force, Seed or Score,
 * by the thresholds of @p view.
 *
 * @param corpus    The facts and their sources.
 * @param view      Which sources to listen to, and where the thresholds sit.
 * @param outReport Receives the tally.
 * @return The timeline, already finalised into its canonical order.
 */
[[nodiscard]] Timeline buildTimeline(const Corpus &corpus, const WorldView &view, FusionReport &outReport);

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_POSSIBLEWORLD_HPP
