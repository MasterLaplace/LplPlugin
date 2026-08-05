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
    core::i32 fromYear{0};   ///< First year the claim covers.
    core::i32 toYear{0};     ///< Last year it covers; equal to @c fromYear for an instant.
    core::u32 source{0u};    ///< Which source asserted it.

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
    Notarial = 0u,     ///< Contracts, registers, acts. Written to be checked.
    Archaeology = 1u,  ///< Material evidence. Silent about motive, hard to forge.
    Administrative = 2u, ///< Censuses, tax rolls. Accurate about what was taxed.
    Chronicle = 3u,    ///< A contemporary account. Honest and partial.
    Panegyric = 4u,    ///< Written to praise. Accurate only by accident.
    Count = 5u
};

/**
 * @struct SourceProfile
 * @brief What is known about a source, as the trust score needs it.
 */
struct SourceProfile {
    core::u32 id{0u};                    ///< Matches Fact::source.
    SourceKind kind{SourceKind::Chronicle};
    core::u32 yearsAfterEvent{0u};       ///< Distance between the event and the writing.
    core::u32 independentAgreements{0u}; ///< Other sources that say the same thing.
};

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
