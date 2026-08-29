/**
 * @file Predicate.hpp
 * @brief What a source can claim, and what an agent can do.
 *
 * @warning **The distinction this file exists for: an ATTRIBUTE is not a DEED.** A source may claim
 * that a king died of dysentery; no agent in a simulation performs "died of dysentery" -- it is a
 * property established after the fact by someone reading bones. But an agent absolutely can
 * travel somewhere, and if it does, that is an event the run produced rather than one the
 * timeline handed it.
 *
 * That line is load-bearing for @ref Divergence, whose entire honesty rests on counting only what
 * a run EARNED: an event the timeline caused cannot count as agreement with that timeline. So the
 * set of predicates an agent may emit has to be smaller than the set a corpus may assert, and
 * bounded here rather than checked at each call site.
 *
 * @warning **The numbering is frozen where it was already taken.** `kPredicateDiedOf = 10` and
 * `kPredicateExists = 11` came from `Parity.hpp` and are folded into gate P13 -- renumbering them
 * would move a signature that six booted artifacts agree on. New predicates take fresh numbers
 * above them; none is ever reused.
 *
 * @warning Predicates only. Subjects and objects are corpus-local identifiers -- a person, a place, an
 * answer -- and belong to whatever curated the corpus. This module trades in identifiers and the
 * strings live in LplKnowledge, which is the rule `Fact.hpp` states.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_PREDICATE_HPP
#    define LPL_LPL_HISTORY_PREDICATE_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::history {

/**
 * @enum Predicate
 * @brief The closed set of things that can be claimed of a subject.
 *
 * Closed, and deliberately small. A vocabulary that grows by guessing produces claims nothing can
 * check; each entry here is either something a corpus states in so many words or something an
 * agent can be seen to do.
 */
enum class Predicate : core::u32 {
    None = 0u,

    // ── Attributes: what a source asserts, and no agent performs ─────────────
    DiedOf = 10u,  ///< What killed him. Frozen: folded into gate P13.
    Exists = 11u,  ///< Whether a settlement is there. Frozen: folded into gate P13.
    BornIn = 12u,  ///< The year of a birth, as a source dates it.
    DiedIn = 13u,  ///< The year of a death.
    Ruled = 14u,   ///< Held authority over a place, across the fact's window.
    Wrote = 15u,   ///< Authored a work. The object is the work.

    // ── Deeds: what an agent can be seen to do, and therefore EARN ───────────
    BornAt = 40u,     ///< Came into the world at a place.
    DweltAt = 41u,    ///< Was living at a place, across the fact's window.
    TravelledTo = 42u,///< Went somewhere. The deed a walking entity produces.
    DiedAt = 43u,     ///< Ended at a place.
    Founded = 44u,    ///< Made a settlement exist that did not.
    Abandoned = 45u,  ///< Left a settlement with nobody in it.

    Count = 46u, ///< One past the highest, for bounds checks. NOT a count of entries.
};

/**
 * @brief Whether a predicate names something an agent can DO.
 *
 * @warning The gate on emergent history. @ref Divergence only credits a run for events it produced on
 * its own, so an agent allowed to emit `DiedOf` could manufacture agreement with a claim it was
 * in no position to establish -- and the measurement would flatter the run for free.
 *
 * @param predicate The predicate.
 * @return true when an agent may emit it.
 */
[[nodiscard]] constexpr bool isDeed(Predicate predicate) noexcept
{
    return predicate == Predicate::BornAt || predicate == Predicate::DweltAt ||
           predicate == Predicate::TravelledTo || predicate == Predicate::DiedAt ||
           predicate == Predicate::Founded || predicate == Predicate::Abandoned;
}

/**
 * @brief Whether a predicate's object is a PLACE rather than a value.
 *
 * The gazetteer answers "where is this identifier"; it must only be asked about objects that are
 * places. Asking it what `DiedOf` points at would resolve dysentery to a coordinate.
 *
 * @param predicate The predicate.
 * @return true when the object names a place.
 */
[[nodiscard]] constexpr bool objectIsPlace(Predicate predicate) noexcept
{
    return predicate == Predicate::BornAt || predicate == Predicate::DweltAt ||
           predicate == Predicate::TravelledTo || predicate == Predicate::DiedAt ||
           predicate == Predicate::Founded || predicate == Predicate::Abandoned ||
           predicate == Predicate::Ruled;
}

/**
 * @brief What a predicate reads as.
 *
 * A word, never an index. An index means whatever the enumeration was worth the day a document
 * was written, so reordering would silently reinterpret everything already on disk -- the rule
 * `WorldRecipe` already keeps for cave kinds.
 *
 * @param predicate The predicate.
 * @return Its name, or "unknown".
 */
[[nodiscard]] constexpr const char *predicateName(Predicate predicate) noexcept
{
    switch (predicate)
    {
    case Predicate::DiedOf:      return "died-of";
    case Predicate::Exists:      return "exists";
    case Predicate::BornIn:      return "born-in";
    case Predicate::DiedIn:      return "died-in";
    case Predicate::Ruled:       return "ruled";
    case Predicate::Wrote:       return "wrote";
    case Predicate::BornAt:      return "born-at";
    case Predicate::DweltAt:     return "dwelt-at";
    case Predicate::TravelledTo: return "travelled-to";
    case Predicate::DiedAt:      return "died-at";
    case Predicate::Founded:     return "founded";
    case Predicate::Abandoned:   return "abandoned";
    case Predicate::None:
    case Predicate::Count:
    default:                     return "unknown";
    }
}

/**
 * @brief Reads a predicate back from its word.
 *
 * @warning An unknown word is REFUSED with its reason rather than defaulted. A document naming a verb
 * this build does not know is a document about something else, and quietly turning it into
 * `None` would file that claim under nothing.
 *
 * @param text  The word.
 * @param bytes Its length.
 * @param out   Receives the predicate.
 * @return false when the word names no predicate.
 */
[[nodiscard]] constexpr bool predicateByName(const char *text, core::u32 bytes, Predicate &out) noexcept
{
    constexpr Predicate kAll[] = {Predicate::DiedOf,  Predicate::Exists,      Predicate::BornIn,
                                  Predicate::DiedIn,  Predicate::Ruled,       Predicate::Wrote,
                                  Predicate::BornAt,  Predicate::DweltAt,     Predicate::TravelledTo,
                                  Predicate::DiedAt,  Predicate::Founded,     Predicate::Abandoned};
    for (const Predicate candidate : kAll)
    {
        const char *name = predicateName(candidate);
        core::u32 i = 0u;
        while (i < bytes && name[i] != '\0' && name[i] == text[i])
            ++i;
        if (i == bytes && name[i] == '\0')
        {
            out = candidate;
            return true;
        }
    }
    return false;
}

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_PREDICATE_HPP
