/**
 * @file Fact.hpp
 * @brief The sextuplet: subject, predicate, object, validity, source, confidence.
 *
 * A historical fact is not true or false, it is asserted by a source over a time
 * window with a confidence. This is the POD form of that statement, sigma held as
 * Fixed32 so a confidence never differs by a rounding between host and kernel.
 * Deliberately flat and versioned: it crosses the wire from LplKnowledge.
 *
 * Subject, predicate and object are IDENTIFIERS, not strings. The strings live in
 * LplKnowledge, which is where a corpus is curated; carrying them here would put a
 * heap and a text encoding into a module that has to run in ring 0, and would make
 * two facts about the same person compare unequal because one spelling has an accent.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_FACT_HPP
#    define LPL_LPL_HISTORY_FACT_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::history {

/**
 * @struct Fact
 * @brief One assertion, by one source, over one window.
 */
struct Fact {
    core::u32 subject{0u};   ///< Who or what the claim is about.
    core::u32 predicate{0u}; ///< What is claimed of it.
    core::u32 object{0u};    ///< The value claimed.
    /**
     * First day the claim covers. See @ref Calendar.hpp for the epoch.
     *
     * @warning DAYS, and the unit carries the precision. This was `fromYear`, so a diary entry dated
     * 18 June 1815 was stored as "1815" and the day thrown away at ingestion -- irreversibly, and
     * identically to a chronicle that only knew the year. Ancient material genuinely is
     * year-resolution and modern material is not; a type that cannot tell them apart makes every
     * consumer treat a ship's log and a legend as equally sharp.
     */
    core::i32 fromDay{0};

    /**
     * Last day it covers; equal to @c fromDay for an instant.
     *
     * @warning The WIDTH of this interval IS the precision, which is why the change needed no new
     * field: "1815" is [1 Jan, 31 Dec] and "18 June 1815" is one day. A separate precision
     * enum would have been a second answer to a question the interval already answers.
     */
    core::i32 toDay{0};

    core::u32 source{0u}; ///< Which source asserted it.

    /**
     * @brief Confidence in [0, 1].
     *
     * Fixed32, never a float: a confidence decides which of two contradictory claims
     * becomes the consensus view, so it is authoritative state and a rounding that
     * differs between targets would give two different histories.
     */
    math::Fixed32 sigma{};
};

/**
 * @enum SourceKind
 * @brief What kind of thing said it.
 *
 * The ordering is not alphabetical and not arbitrary: it runs from the sources whose
 * interest is to record accurately to the ones whose interest is to persuade. A
 * notarial act was written to settle a dispute later; a panegyric was written to
 * flatter someone alive at the time.
 */
enum class SourceKind : core::u32 {
    Notarial = 0u,       ///< Contracts, registers, acts. Written to be checked.
    Archaeology = 1u,    ///< Material evidence. Silent about motive, hard to forge.
    Administrative = 2u, ///< Censuses, tax rolls. Accurate about what was taxed.
    Chronicle = 3u,      ///< A contemporary account. Honest and partial.
    Panegyric = 4u,      ///< Written to praise. Accurate only by accident.
    Count = 5u
};

/**
 * Distance nobody has established.
 *
 * @warning **A sentinel taken from OUTSIDE the range, because zero already means something.** Zero
 * years after the event is an eyewitness -- the strongest thing a source can be -- so using it
 * for "we do not know" hands full temporal credit to every source whose composition date was
 * never recorded. That is not a hypothetical: every TEI-ingested work wrote zero, so Herodotus
 * writing three generations after Croesus was scored as though he had been standing there.
 *
 * Same shape as `kAnyLanguage` and `kNoIdentifier` elsewhere in this project, and the same
 * lesson: a field whose whole range is meaningful needs its "unknown" from outside it.
 */
inline constexpr core::u32 kUnknownYearsAfterEvent = 0xFFFFFFFFu;

/**
 * @struct SourceProfile
 * @brief What is known about a source, as the trust score needs it.
 */
struct SourceProfile {
    core::u32 id{0u}; ///< Matches Fact::source.
    SourceKind kind{SourceKind::Chronicle};

    /**
     * Years between the event and the writing, or @ref kUnknownYearsAfterEvent.
     *
     * @warning **This depends on the FACT, not only on the source, which is why
     * @ref distanceInYears exists.** A chronicle compiled in 1200 that recounts both 1190 and 400
     * is ten years from one claim and eight hundred from the other; a single number on the source
     * cannot be right about both. Set it directly only for a source dated relative to its subject
     * and nothing else -- a report written as its own research happened, say. Otherwise give
     * @ref composedFrom and let the distance be derived per claim.
     */
    core::u32 yearsAfterEvent{kUnknownYearsAfterEvent};

    /**
     * First day the source could have been written; 0 when unknown.
     *
     * An interval, like everything else dated here: "the 420s BCE" is what is known about
     * Herodotus, and a single day would claim a precision no one has.
     */
    core::i32 composedFrom{0};

    /// Last day it could have been written; 0 when unknown.
    core::i32 composedTo{0};

    core::u32 independentAgreements{0u}; ///< Other sources that say the same thing.
};

/**
 * @brief How long after an event a source was written.
 *
 * @warning Derived from the composition window when there is one, because the answer belongs to the
 * PAIR rather than to the source. Falls back to a directly stated distance, and answers
 * @ref kUnknownYearsAfterEvent when neither is known -- which the trust score must then treat as
 * a credit not earned rather than as a credit granted.
 *
 * @warning **The gap to the nearest edge PLUS the width of the window**, and both halves are needed.
 * The gap alone gives a source the benefit of its own vagueness: widen a window until the event
 * falls inside it and the gap is zero, so an undatable manuscript scores as an eyewitness.
 * Adding the width makes widening cost something, which is what it should cost -- not knowing
 * when a text was written to within fifty years is a reason to trust it less about a particular
 * year, and a precisely dated source keeps its full credit.
 *
 * @param profile  The source.
 * @param eventDay The day the claimed event happened.
 * @return The distance in years, or @ref kUnknownYearsAfterEvent.
 */
[[nodiscard]] core::u32 distanceInYears(const SourceProfile &profile, core::i32 eventDay) noexcept;

/**
 * @struct TrustWeights
 * @brief The three terms of the trust score, and what each is worth.
 *
 * Weights rather than a formula baked in: which of the three matters most is a
 * historiographical position, not a fact, and a project that hardcodes it has taken
 * that position without saying so.
 */
struct TrustWeights {
    math::Fixed32 temporalProximity{math::Fixed32::fromRaw(19661)}; ///< ~0.30
    math::Fixed32 sourceType{math::Fixed32::fromRaw(29491)};        ///< ~0.45
    math::Fixed32 peerConsensus{math::Fixed32::fromRaw(16384)};     ///< ~0.25
};

/**
 * @brief How much a source is worth believing, in [0, 1].
 *
 * R(S) = w1 * temporal proximity + w2 * source type + w3 * peer consensus.
 *
 * Temporal proximity decays with distance rather than falling off a cliff: an account
 * written five years after an event is worth much more than one written fifty, and one
 * written five hundred is not worth appreciably less than one written five thousand.
 *
 * @param profile What is known about the source.
 * @param weights What each term is worth.
 * @return The score, clamped to [0, 1].
 */
[[nodiscard]] math::Fixed32 trustworthiness(const SourceProfile &profile,
                                            const TrustWeights &weights = TrustWeights{}) noexcept;

/**
 * @brief How much a source is worth believing ABOUT ONE CLAIM, in [0, 1].
 *
 * @warning **The overload that exists because the distance belongs to the pair.** A chronicle
 * compiled in 1200 recounting both 1190 and 400 is ten years from one claim and eight hundred
 * from the other; scoring it once, for the source as a whole, is right about at most one of them.
 * Prefer this wherever a fact is in hand -- which is everywhere the score is actually used.
 *
 * @param profile  What is known about the source.
 * @param eventDay The day the claimed event happened.
 * @param weights  Which of the three terms matters most.
 * @return The score.
 */
[[nodiscard]] math::Fixed32 trustworthiness(const SourceProfile &profile, core::i32 eventDay,
                                            const TrustWeights &weights = TrustWeights{}) noexcept;

/**
 * @brief Combines two independent confidences in the same claim.
 *
 * P(E | S1, S2) = 1 - (1 - P1)(1 - P2). Ten independent sources that agree raise the
 * confidence; the word doing the work is INDEPENDENT, and this function cannot check
 * it. Two chroniclers copying the same lost original are one source, and fusing them
 * as two is the commonest way a corpus manufactures certainty it has not earned.
 *
 * @param a First confidence.
 * @param b Second confidence.
 * @return The fused confidence, in [0, 1].
 */
[[nodiscard]] math::Fixed32 fuseConfidence(math::Fixed32 a, math::Fixed32 b) noexcept;

/**
 * @brief Do two facts assert incompatible things about the same subject at the same time?
 *
 * The mutual-exclusion rule: same subject, same predicate, overlapping windows,
 * different objects. A person is not in two places in the same year.
 *
 * @param a First fact.
 * @param b Second fact.
 * @return true when they cannot both hold.
 */
[[nodiscard]] bool contradicts(const Fact &a, const Fact &b) noexcept;

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_FACT_HPP
